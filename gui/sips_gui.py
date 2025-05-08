import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import ttk
from tkinter.filedialog import askdirectory, asksaveasfilename, askopenfilename
import numpy as np
import time, sys, io, os, math, cv2, tomllib
from source import series_handling_functions as shf
from source import nielsen_algorithm
from skimage import filters
from tqdm import tqdm
from tifffile import imwrite
from pathlib import PurePath


class SIPS:

    def __init__(self, image_array: np.ndarray, fps: int, pixel_size: float, file_name='File_name',
                 flat: np.ndarray = None, dark: np.ndarray = None, ui_config_file=None, icon_file=None):

        """
        A program for pre- and post processing of 3D greyscale image stacks. Tkinter is used to create an interactive
        GUI for the image manipulation option. Data can be loaded as a 3D numpy array upon call of this class, or
        imported using the import options in the GUI. The following features are available:

        - Post-processing:
            - Clipping: Adjust brightness and contrast levels and removes pixel outliers, similar to ImageJ
            - Gaussian Smoothing: Removes data spikes at the cost of sharpness
            - Non-Local Means Denoising: Removes noise from images (can reduce sharpness when too aggressive)
            - Contrast Limited Adaptive Histogram Equalization: Enhances contrast of low contrast regions
            - Resize: Scales image to fill the screen (down or upscale)
            - Zoom: Allows to zoom into image at a specified point
        - Pre-processing:
            - subtraction of neighbouring frames
            - division of neighbouring frames (ratioing)
        - Saving:
            - Save displayed
            - Save defined range (or all)
            - Export as video
            - Export as binary (.npy)
        - Import and corrections:
            - Import of data from .cine or .npy
            - Import of flat and dark field data for correction
            - Flat and dark field image correction
            - Image cropping/transformation (size adjustment) --> Mark coordinates for cropping and rotation:
              Allows to determine image boundaries and tilt for image transformation
        - Play: Allows to play frames as a video (calculations are applied live)
        - Bake (classmethod): Allows to bake current settings into image_stack (for continuous processing outside of GUI)
        .

        :param image_array:
               3D numpy array of shape: height x width x number of images.
        :param fps:
               Number of frames per second with which the image stack (from a video) was recorded
        :param pixel_size:
               Detector pixel size to compute field of view of image (given in µm)
        :param file_name:
               File name of the currently loaded file (automatically set if loaded via import)
        :param flat:
               2D numpy array of shape: height x width, holding a flat field image for correction
        :param dark:
               2D numpy array of shape: height x width, holding a flat field image for correction
        """


        ################################ Store UI element variables
        # path to folder where images can be saved
        self.source_path = None
        # define default values for sliders and store variables
        self.clip_min_default = 1
        self.clip_max_default = 99
        self.delta_scaler_default = 1
        self.ratio_scaler_default = 1
        self.gaussian_sigma_default = 0.5
        self.nlmeans_template_window_size_default = 7
        self.nlmeans_search_window_size_default = 21
        self.nlmeans_h_slider_default = 2.0
        self.nlmeans_h_default = int(self.nlmeans_h_slider_default / 100 * 65535)
        self.clahe_clip_limit_default = 2.0
        self.clahe_tile_grid_size_default = 8
        self.current_image_number_default = 0
        self.neighbour_order_default = 1
        self.neighbour_order_value = self.neighbour_order_default

        ############################## Store important data
        # hold an image array for corrections
        current_directory = os.path.dirname(__file__)
        self.image_array = image_array
        self.displayed_array = self.image_array
        self.fps = fps
        self.pixel_size = pixel_size
        self.file_name = file_name
        # hold loaded flat and dark field arrays for correction
        self.flat = flat
        self.dark = dark
        # load first image of stack to retrieve image size
        self.current_image_number = self.current_image_number_default
        # variable storing the currently displayed image as an 8-bit numpy array
        self.currently_displayed_image = None

        ############################# Retrieve and store important image size information for GUI window creation
        # globally define the image width and height (used to determine real image size)
        # also define the shift of the image coordinate system to the canvas coordinate system to make sure points are
        # positioned correctly
        self.image_width = self.displayed_array[:, :, 0].shape[1]
        self.image_height = self.displayed_array[:, :, 0].shape[0]
        self.number_of_images = self.displayed_array.shape[2]

        # some additional variables that are useful to retrieve from class
        self.x_left = 0
        self.x_right = self.image_width
        self.y_bot = self.image_height
        self.y_top = 0
        self.image_tilt = 0
        self.x_centre_zoom = self.image_width // 2
        self.y_centre_zoom = self.image_height // 2

        ##################################### Determine appropriate GUI window and canvas size
        try:
            # tries to load the UI's config file and retrieve display settings
            if ui_config_file:
                f = open(ui_config_file, 'rb')
            else:
                f = open('./run/sips_config.toml', 'rb')
            config = tomllib.load(f)
            f.close()
            self.font_scaling_factor = config['display_settings']['font_scaling_factor']
            self.window_width = config['display_settings']['window_width']
            self.window_height = config['display_settings']['window_height']
            self.auto_image_resize = config['display_settings']['enable_auto_image_resize']
        except (tomllib.TOMLDecodeError, FileNotFoundError) as error:
            self.font_scaling_factor = 1.0    # factor by which UI font is scaled (does not affect UI window size)
            self.window_width = 1920          # UI window with
            self.window_height = 1080         # UI window height
            self.auto_image_resize = True     # enables or disables auto image resizing on UI window size changes

        # default values of the UI size that lead to good font/UI element proportions
        self.default_window_width = 1920   # do not touch this value (reference for UI to font proportions) !!!
        self.default_window_height = 1080  # do not touch this value (reference for UI to font proportions) !!!
        self.default_font_size = 11        # do not touch this value (reference for UI to font proportions) !!!
        self.maximum_font_size = 14        # limits the font size to this value
        
        # rescale widget font based on difference between default window size and selected window size
        self.font_auto_scaler = min(self.window_width / self.default_window_width, self.window_height / self.default_window_height)
        self.fontsetting = ("Calibri", min(math.floor(self.font_scaling_factor * self.font_auto_scaler * self.default_font_size), 14))
        # determine correction factor for proper centering of image in canvas
        self.canvas_width = int(2 * self.window_width / 3)  # proportional canvas width
        self.canvas_height = int(5 * self.window_height / 6)  # proportional canvas height
        self.x_shift = (self.canvas_width - self.image_width) // 2
        self.y_shift = (self.canvas_height - self.image_height) // 2

        ####################################### Construct GUI window
        # self.root = tk.Tk()
        self.root = ThemedTk(theme='breeze')
        try:
            # try to import an icon file for the UI
            if icon_file:
                self.root.iconphoto(False, tk.PhotoImage(file=PurePath(icon_file)))
            else:
                self.root.iconphoto(False, tk.PhotoImage(file=PurePath(current_directory, '../run/SIPS_icon.png')))
                print("Check icon")
        except FileNotFoundError:
            pass
        self.root.title("Synchrotron Imaging Processing Suite (SIPS)" + f" - File: {self.file_name}     " +
                        f"(Image size: {int(self.image_width * self.pixel_size)} µm x {int(self.image_height * self.pixel_size)} µm)")
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')
        self.root.grid_rowconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')

        ###################################### Create frames that hold the UI elements

        # frame that holds the canvas on which image is displayed
        self.cframe = tk.Frame(self.root)
        self.cframe.grid(row=0, column=0, rowspan=5, columnspan=4, sticky="nsew")
        self.cframe.grid_columnconfigure(0, weight=1)
        self.cframe.grid_rowconfigure(0, weight=1)
        # canvas on which image is displayed
        self.canvas = tk.Canvas(self.cframe, bg='grey80')
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.cframe.grid_propagate(False)

        # frame that contains the widgets that are sorted into tabs
        self.tframe = tk.Frame(self.root)
        self.tframe.grid(row=0, column=4, rowspan=5, columnspan=2, sticky="nsew")
        self.tframe.grid_columnconfigure(0, weight=1)
        self.tframe.grid_rowconfigure(0, weight=1)
        self.tframe.grid_propagate(False)   # prevents Notebook from increasing frame size beyond its limit

        # frame that holds the global widgets below the canvas
        self.bcframe = tk.Frame(self.root)
        self.bcframe.grid(row=5, column=0, columnspan=4, sticky="nsew")
        self.bcframe.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1, uniform='a')
        self.bcframe.grid_rowconfigure((0, 1, 2), weight=1, uniform='a')
        self.bcframe.grid_propagate(False)

        # frame that holds the global widgets below the Tabs
        self.btframe = tk.Frame(self.root)
        self.btframe.grid(row=5, column=4, rowspan=1, columnspan=2, sticky="nsew")
        self.btframe.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')
        self.btframe.grid_rowconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')
        self.btframe.grid_propagate(False)

        # create tabs for the tab frame
        self.style = ttk.Style()
        self.style.configure('.', font=self.fontsetting)
        # notebook that holds the tabs
        self.tabControl = ttk.Notebook(self.tframe)
        self.tabControl.grid(row=0, column=0, sticky="nsew")
        self.tabControl.grid_columnconfigure(0, weight=1)
        self.tabControl.grid_rowconfigure(0, weight=1)
        # individual tabs
        self.tab1 = tk.Frame(self.tabControl)
        self.tab2 = tk.Frame(self.tabControl)
        self.tab3 = tk.Frame(self.tabControl)
        self.tab4 = tk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text='View settings')
        self.tabControl.add(self.tab2, text='Processing')
        self.tabControl.add(self.tab3, text='Save data')
        self.tabControl.add(self.tab4, text='Load data')
        # tab size definitions
        self.tab1.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')
        self.tab1.grid_rowconfigure(list(range(0, 23)), weight=1)
        self.tab2.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')
        self.tab2.grid_rowconfigure(list(range(0, 23)), weight=1)
        self.tab3.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')
        self.tab3.grid_rowconfigure(list(range(0, 23)), weight=1)
        self.tab4.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform='a')
        self.tab4.grid_rowconfigure(list(range(0, 23)), weight=1)

        # Mouse and Keyboard events that interact with UI (e.g. placement of coordinate markers)
        self.root.bind("<Enter>", self.on_enter)
        self.root.bind("<Leave>", self.on_leave)
        self.root.bind("<Configure>", self.resizing)
        self.canvas.bind('<Button-1>', self.callback_left)
        self.canvas.bind('<Button-3>', self.callback_right)
        self.canvas.bind('<Shift-Button-1>', self.callback_angle)
        self.canvas.bind('<Shift-Button-3>', self.callback_zoom_centre)
        self.clicked = False
        self.angle_coordinates = {}

        ################################################### BC Widgets #################################################
        ################################################################################################################
        ################################################################################################################

        # create a slider to scroll through image stack
        self.current_image_scale = tk.Scale(master=self.bcframe, orient="horizontal", from_=0, font=self.fontsetting,
                                            to=self.displayed_array.shape[2] - 1, activebackground='green', resolution=1,
                                            showvalue=True, command=self.display_next_image, sliderlength=50)

        # define text labels to display information from mouse button presses
        self.coordinate_a = tk.Label(self.bcframe, text=f"L-Click: A = ({self.x_left}, {self.y_top}); top-left",
                                     background='red', font=self.fontsetting, relief='flat')
        self.coordinate_b = tk.Label(self.bcframe, text=f"R-Click: B = ({self.x_right}, {self.y_bot}); bot-right",
                                     background='green', font=self.fontsetting, relief='flat')
        self.angle_measurement = tk.Label(self.bcframe, text=f"Shift + L-Click: Set two points to measure angle",
                                          background='yellow', font=self.fontsetting, relief='flat')

        # create a text time_label that displays time increments counting from first frame of the stack
        self.time_label = tk.Label(self.bcframe, text='Time passed in ms.', width=10, anchor='w', padx=5,
                                   background='black', font=self.fontsetting, fg="white")
        self.time_increment = 1000 / self.fps  # ms

        # Calculation widgets based on red and green marker position
        self.distance_ab_value = int(
            np.sqrt((self.x_right - self.x_left) ** 2 + (self.y_top - self.y_bot) ** 2) * self.pixel_size)
        self.distance_ab_label = tk.Label(self.bcframe, text=f"A - B distance = {self.distance_ab_value} µm",
                                          background='grey', font=self.fontsetting, relief='flat')
        self.velocity_ab_value = int(self.distance_ab_value / (1000 * self.neighbour_order_value / self.fps))
        self.velocity_ab_label = tk.Label(self.bcframe, text=f"A - B velocity = {self.velocity_ab_value} mm/s",
                                          background='grey', font=self.fontsetting, relief='flat')

        self.previous_frame_button = tk.Button(self.bcframe, text="<-- Previous", bg='grey80', command=self.previous_frame,
                                               font=self.fontsetting, relief='groove', border=3)
        self.next_frame_button = tk.Button(self.bcframe, text="Next -->", bg='grey80', command=self.next_frame,
                                           font=self.fontsetting, relief='groove', border=3)

        # Widget placement
        self.coordinate_a.grid(column=0, columnspan=2, row=0, sticky='nsew', padx=2, pady=2)
        self.coordinate_b.grid(column=2, columnspan=2, row=0, sticky='nsew', padx=2, pady=2)
        self.angle_measurement.grid(column=4, columnspan=3, row=0, sticky='nsew', padx=2, pady=2)
        self.time_label.grid(column=7, row=0, sticky='nsew', padx=2, pady=2)
        self.distance_ab_label.grid(column=0, columnspan=2, row=1, sticky='nsew', padx=2, pady=2)
        self.velocity_ab_label.grid(column=2, columnspan=2, row=1, sticky='nsew', padx=2, pady=2)
        self.current_image_scale.grid(column=0, row=2, columnspan=8, sticky='nsew', padx=2, pady=2)

        self.previous_frame_button.grid(column=4, columnspan=2, row=1, sticky='nsew', padx=2, pady=2)
        self.next_frame_button.grid(column=6, columnspan=2, row=1, sticky='nsew', padx=2, pady=2)

        ################################################### BT Widgets #################################################
        ################################################################################################################
        ################################################################################################################

        # widget to play the frames as a video
        self.play_state = False
        self.image_skip_value = tk.StringVar()
        self.play_pause_button = tk.Button(self.btframe, text="PLAY", bg='green', command=self.start_play_video,
                                           font=self.fontsetting, relief='groove', border=3)
        self.play_pause_image_skip_entry = tk.Entry(self.btframe, textvariable=self.image_skip_value, font=self.fontsetting,
                                                    validate='key')
        self.play_pause_image_skip_entry['validatecommand'] = (self.play_pause_image_skip_entry.register(shf.testVal), '%P', '%d')
        self.play_pause_image_skip_entry.insert(0, "1")
        self.play_pause_image_skip_label = tk.Label(self.btframe, text="frames skipped", font=self.fontsetting)

        # play from to widgets
        self.play_from_value = tk.StringVar()
        self.play_from_label = tk.Label(self.btframe, text=f"from:", font=self.fontsetting, bg='grey')
        self.play_from_entry = tk.Entry(self.btframe, textvariable=self.play_from_value, font=self.fontsetting, validate='key')
        self.play_from_entry['validatecommand'] = (self.play_from_entry.register(shf.testVal), '%P', '%d')
        self.play_from_entry.insert(0, "0")
        self.play_from = int(self.play_from_value.get())

        self.play_to_label = tk.Label(self.btframe, text=f"to:", font=self.fontsetting, bg='grey')
        self.play_to_value = tk.StringVar()
        self.play_to_entry = tk.Entry(self.btframe, textvariable=self.play_to_value, font=self.fontsetting, validate='key')
        self.play_to_entry['validatecommand'] = (self.play_to_entry.register(shf.testVal), '%P', '%d')
        self.play_to_entry.insert(0, str(self.number_of_images - 1))
        self.play_to = int(self.play_to_value.get())

        # widget to reset all settings to default state
        self.reset_button = tk.Button(self.btframe, text="Reset View settings", bg='grey', command=self.reset_settings,
                                      font=self.fontsetting, relief='groove', border=3)

        # Widget placement
        self.play_pause_button.grid(row=0, rowspan=3, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.play_pause_image_skip_entry.grid(row=2, column=3, columnspan=1, sticky='sew', padx=2, pady=2)
        self.play_pause_image_skip_label.grid(row=2, column=4, columnspan=2, sticky='sew', padx=2, pady=2)
        self.play_from_label.grid(row=3, column=0, sticky='sew', padx=2, pady=2)
        self.play_from_entry.grid(row=3, column=1, sticky='sew', padx=2, pady=2)
        self.play_to_label.grid(row=3, column=2, sticky='sew', padx=2, pady=2)
        self.play_to_entry.grid(row=3, column=3, sticky='sew', padx=2, pady=2)

        self.reset_button.grid(row=4, rowspan=2, column=0, columnspan=6, sticky='nsew', padx=2, pady=2)

        ######################################### Tab 1 Widgets ########################################################
        ################################################################################################################
        ################################################################################################################

        cb_h = 2    # checkbutton height
        label_h = 1 # label height

        # widgets for clipping adjustment
        self.clipping_check = tk.IntVar()
        self.clipping_checkbutton = tk.Checkbutton(self.tab1, height=cb_h, text="Clipping", variable=self.clipping_check,
                                                   bg='grey', command=self.redraw_image, font=self.fontsetting)
        self.clip_min_value = self.clip_min_default
        self.clip_min_label = tk.Label(self.tab1, height=label_h, text=f"min %:", font=self.fontsetting)
        self.clip_min_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=0, to=20, resolution=0.1, showvalue=True,
                                       command=self.adjust_clip_min_value, sliderlength=30, activebackground='green', font=self.fontsetting)
        self.clip_min_scale.set(self.clip_min_value)
        self.clip_max_value = self.clip_max_default
        self.clip_max_label = tk.Label(self.tab1, height=label_h, text=f"max %:", font=self.fontsetting)
        self.clip_max_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=80, to=100, resolution=0.1, showvalue=True,
                                       command=self.adjust_clip_max_value, sliderlength=30, activebackground='green', font=self.fontsetting)
        self.clip_max_scale.set(self.clip_max_value)

        # widgets for gaussian smoothing
        self.gaussian_check = tk.IntVar()
        self.gaussian_checkbutton = tk.Checkbutton(self.tab1, height=cb_h, text="Gaussian Smoothing",font=self.fontsetting,
                                                   variable=self.gaussian_check, bg='grey', command=self.redraw_image)
        self.gaussian_sigma_value = self.gaussian_sigma_default
        self.gaussian_sigma_label = tk.Label(self.tab1, height=label_h, text=f"sigma:", font=self.fontsetting)
        self.gaussian_sigma_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=0, to=10, resolution=0.1,
                                             showvalue=True, font=self.fontsetting, activebackground='green',
                                             command=self.adjust_gaussian_sigma_value, sliderlength=30)
        self.gaussian_sigma_scale.set(self.gaussian_sigma_value)

        # widgets for fast non-local means denoising
        self.nlmeans_check = tk.IntVar()
        self.nlmeans_checkbutton = tk.Checkbutton(self.tab1, height=cb_h, text="Fast Non-Local Means Denoising (Slow!)",
                                                  variable=self.nlmeans_check, bg='grey', command=self.redraw_image,
                                                  font=self.fontsetting)
        self.nlmeans_template_window_size_value = self.nlmeans_template_window_size_default
        self.nlmeans_template_window_size_label = tk.Label(self.tab1, height=label_h, text=f"templateWindowSize:",
                                                           font=self.fontsetting)
        self.nlmeans_template_window_size_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=1, to=30,
                                                           resolution=1, showvalue=True, font=self.fontsetting,
                                                           command=self.adjust_nlmeans_template_window_size_value,
                                                           sliderlength=30, activebackground='green')
        self.nlmeans_template_window_size_scale.set(self.nlmeans_template_window_size_value)
        self.nlmeans_search_window_size_value = self.nlmeans_search_window_size_default
        self.nlmeans_search_window_size_label = tk.Label(self.tab1, height=label_h, text=f"searchWindowSize:",
                                                         font=self.fontsetting)
        self.nlmeans_search_window_size_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=1, to=50,
                                                         resolution=2, font=self.fontsetting, showvalue=True,
                                                         command=self.adjust_nlmeans_search_window_size_value,
                                                         sliderlength=30, activebackground='green')
        self.nlmeans_search_window_size_scale.set(self.nlmeans_search_window_size_value)
        self.nlmeans_h_value = self.nlmeans_h_default
        self.nlmeans_h_slider_value = self.nlmeans_h_slider_default
        self.nlmeans_h_label = tk.Label(self.tab1, height=label_h, text=f"h (% grey-range):", font=self.fontsetting)
        self.nlmeans_h_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=0, to=20, resolution=0.1,
                                        showvalue=True, font=self.fontsetting,
                                        command=self.adjust_nlmeans_h_value, sliderlength=30, activebackground='green')
        self.nlmeans_h_scale.set(self.nlmeans_h_slider_value)

        # widgets for CLAHE (contrast limited adaptive histogram equalisation)
        self.clahe_check = tk.IntVar()
        self.clahe_checkbutton = tk.Checkbutton(self.tab1, height=cb_h,
                                                text="Contrast Limited Adaptive Histogram Equalization",
                                                variable=self.clahe_check, bg='grey', command=self.redraw_image,
                                                font=self.fontsetting)
        self.clahe_clip_limit_value = self.clahe_clip_limit_default
        self.clahe_clip_limit_label = tk.Label(self.tab1, height=label_h, text=f"clip limit:", font=self.fontsetting)
        self.clahe_clip_limit_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=0, to=100, resolution=0.1,
                                               showvalue=True, command=self.adjust_clahe_clip_limit_value,
                                               sliderlength=30, font=self.fontsetting,
                                               activebackground='green')
        self.clahe_clip_limit_scale.set(self.clahe_clip_limit_value)
        self.clahe_tile_grid_size_value = (self.clahe_tile_grid_size_default, self.clahe_tile_grid_size_default)
        self.clahe_tile_grid_size_label = tk.Label(self.tab1, height=label_h, text=f"tile grid size:", font=self.fontsetting)
        self.clahe_tile_grid_size_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=1, to=20, resolution=1,
                                                   showvalue=True, command=self.adjust_tile_grid_size_value,
                                                   sliderlength=30, activebackground='green', font=self.fontsetting)
        self.clahe_tile_grid_size_scale.set(self.clahe_tile_grid_size_value[0])

        # widgets for adjusting image current_image_scale/zoom

        if self.auto_image_resize is True:
            # initiated automatically to make image fill canvas
            self.resize_value = self.image_data_canvas_compare()
        else:
            self.resize_value = 1
        self.resize_label = tk.Label(self.tab1, height=2, text="Resize:", bg='grey', font=self.fontsetting)
        self.resize_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=0.3, to=3, resolution=0.01,
                                     showvalue=True, font=self.fontsetting, bg='grey', activebackground='green',
                                     command=self.adjust_resize_level, sliderlength=30)
        self.resize_scale.set(self.resize_value)
        self.zoom_value = 1
        self.zoom_label = tk.Label(self.tab1, height=2, text="Zoom:", bg='grey', font=self.fontsetting)
        self.zoom_scale = tk.Scale(master=self.tab1, orient="horizontal", from_=1, to=5, resolution=0.1, showvalue=True,
                                   command=self.adjust_zoom_level, sliderlength=30, activebackground='green', bg='grey',
                                   font=self.fontsetting)
        self.zoom_scale.set(1)
        self.zoom_centre_button = tk.Button(self.tab1, height=4, background='cyan', font=self.fontsetting,
                                            text=f"Zoom centre:\n Shift + R-Click\n to select\n(Press to reset)",
                                            command=self.reset_zoom_settings, relief='groove', border=3)

        # Widget placement
        self.clipping_checkbutton.grid(row=0, column=0, columnspan=6, sticky='nsew', pady=2)
        self.clip_min_label.grid(row=2, column=0, columnspan=2, sticky='sew', pady=2)
        self.clip_min_scale.grid(row=1, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)
        self.clip_max_label.grid(row=4, column=0, columnspan=2, sticky='sew', pady=2)
        self.clip_max_scale.grid(row=3, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)

        self.gaussian_checkbutton.grid(row=5, column=0, columnspan=6, sticky='nsew', pady=2)
        self.gaussian_sigma_label.grid(row=7, column=0, columnspan=2, sticky='sew', pady=2)
        self.gaussian_sigma_scale.grid(row=6, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)

        self.nlmeans_checkbutton.grid(row=8, column=0, columnspan=6, sticky='nsew', pady=2)
        self.nlmeans_template_window_size_label.grid(row=10, column=0, columnspan=2, sticky='sew', pady=2)
        self.nlmeans_template_window_size_scale.grid(row=9, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)
        self.nlmeans_search_window_size_label.grid(row=12, column=0, columnspan=2, sticky='sew', pady=2)
        self.nlmeans_search_window_size_scale.grid(row=11, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)
        self.nlmeans_h_label.grid(row=14, column=0, columnspan=2, sticky='sew', pady=2)
        self.nlmeans_h_scale.grid(row=13, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)

        self.clahe_checkbutton.grid(row=15, column=0, columnspan=6, sticky='nsew', pady=2)
        self.clahe_clip_limit_label.grid(row=17, column=0, columnspan=2, sticky='sew', pady=2)
        self.clahe_clip_limit_scale.grid(row=16, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)
        self.clahe_tile_grid_size_label.grid(row=19, column=0, columnspan=2, sticky='sew', pady=2)
        self.clahe_tile_grid_size_scale.grid(row=18, column=2, rowspan=2, columnspan=4, sticky='sew', pady=2)

        self.resize_label.grid(row=20, column=0, rowspan=2, columnspan=1, sticky='nsew', padx=2, pady=2)
        self.resize_scale.grid(row=20, column=1, rowspan=2, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.zoom_label.grid(row=22, column=0, rowspan=2, columnspan=1, sticky='nsew', padx=2, pady=2)
        self.zoom_scale.grid(row=22, column=1, rowspan=2, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.zoom_centre_button.grid(row=20, column=4, rowspan=4, columnspan=2, sticky='nsew', padx=2)

        ################################################# Tab 2 Widgets ################################################
        ################################################################################################################
        ################################################################################################################

        # Basic image arithmetic widgets
        self.pre_processing_default = 0
        self.pre_processing_value = self.pre_processing_default
        pre_processing_labels = f"<-- Mode/Distance btw. images -->"
        self.pre_processing_label = tk.Label(self.tab2, text=pre_processing_labels, font=self.fontsetting, bg='grey')
        self.pre_processing_scale = tk.Scale(master=self.tab2, orient="vertical", from_=0, to=5, resolution=1,
                                             showvalue=False, command=self.select_pre_processing_mode, sliderlength=30,
                                             activebackground='green', font=self.fontsetting)
        self.pre_processing_scale.set(self.pre_processing_default)
        self.mode_0_label = tk.Label(self.tab2, text=f"Standard", font=self.fontsetting, bg='grey')
        self.mode_1_label = tk.Label(self.tab2, text=f"Delta total", font=self.fontsetting, bg='grey')
        self.mode_2_label = tk.Label(self.tab2, text=f"Delta neighbours -->", font=self.fontsetting, bg='grey')
        self.mode_3_label = tk.Label(self.tab2, text=f"Ratio total", font=self.fontsetting, bg='grey')
        self.mode_4_label = tk.Label(self.tab2, text=f"Ratio neighbours -->", font=self.fontsetting, bg='grey')
        self.mode_5_label = tk.Label(self.tab2, text=f"Nielsen Optim. -->", font=self.fontsetting, bg='grey')

        self.neighbour_order_scale = tk.Scale(master=self.tab2, orient="vertical", from_=1, to=10, resolution=1,
                                              showvalue=True, command=self.select_neighbour_order, sliderlength=30,
                                              activebackground='green', font=self.fontsetting)
        self.neighbour_order_scale.set(self.neighbour_order_default)

        self.nielsen_optimisation_label = tk.Label(self.tab2, text=f"Nielsen optimisation", font=self.fontsetting, bg='grey')
        self.delta_scaler_value = self.delta_scaler_default
        self.delta_scaler_label = tk.Label(self.tab2, text=f"Delta scaling factor:", font=self.fontsetting)

        self.delta_scaler_scale = tk.Scale(master=self.tab2, orient="horizontal", from_=1, to=200, resolution=1,
                                           showvalue=True, font=self.fontsetting, sliderlength=30,
                                           activebackground='green', command=self.adjust_delta_scaler_value)
        self.delta_scaler_scale.set(self.delta_scaler_value)

        self.ratio_scaler_value = self.ratio_scaler_default
        self.ratio_scaler_label = tk.Label(self.tab2, text=f"Ratio scaling factor:", font=self.fontsetting)
        self.ratio_scaler_scale = tk.Scale(master=self.tab2, orient="horizontal", from_=1, to=200, resolution=1,
                                           showvalue=True, font=self.fontsetting, activebackground='green',
                                           command=self.adjust_ratio_scaler_value, sliderlength=30)
        self.ratio_scaler_scale.set(self.ratio_scaler_value)

        # Widget placement
        self.pre_processing_label.grid(row=0, column=0, columnspan=6, sticky='nsew', pady=2)

        self.pre_processing_scale.grid(row=1, column=0, columnspan=1, rowspan=6, sticky='nse', padx=(0, 20), pady=2)
        self.mode_0_label.grid(row=1, column=1, columnspan=4, sticky='nsew', pady=2, padx=2)
        self.mode_1_label.grid(row=2, column=1, columnspan=4, sticky='nsew', pady=2, padx=2)
        self.mode_2_label.grid(row=3, column=1, columnspan=4, sticky='nsew', pady=2, padx=2)
        self.mode_3_label.grid(row=4, column=1, columnspan=4, sticky='nsew', pady=2, padx=2)
        self.mode_4_label.grid(row=5, column=1, columnspan=4, sticky='nsew', pady=2, padx=2)
        self.mode_5_label.grid(row=6, column=1, columnspan=4, sticky='nsew', pady=2, padx=2)
        self.neighbour_order_scale.grid(row=1, column=5, columnspan=2, rowspan=6, sticky='nsew', pady=2)

        self.nielsen_optimisation_label.grid(row=7, column=0, columnspan=6, sticky='nsew', pady=(20, 2))
        self.delta_scaler_label.grid(row=9, column=0, columnspan=2, sticky='sew', pady=2)
        self.delta_scaler_scale.grid(row=8, rowspan=2, column=2, columnspan=4, sticky='sew', pady=2)
        self.ratio_scaler_label.grid(row=11, column=0, columnspan=2, sticky='sew', pady=2)
        self.ratio_scaler_scale.grid(row=10, rowspan=2, column=2, columnspan=4, sticky='sew', pady=2)

        ################################################# Tab 3 Widgets ################################################
        ################################################################################################################
        ################################################################################################################

        # set a parent directory that the save prompt starts in
        self.parent_directory_value = tk.StringVar()
        self.parent_directory_label = tk.Label(self.tab3, text=f"Export directory (optional):", font=self.fontsetting,
                                               bg='grey')
        self.parent_directory_entry = tk.Entry(self.tab3, textvariable=self.parent_directory_value, font=self.fontsetting)
        self.parent_directory_entry.insert(0, "enter parent directory here")

        # Section with advanced saving
        self.simple_saving_label = tk.Label(self.tab3, text=f"Basic Export", font=self.fontsetting, bg='grey')

        # widget to save currently displayed image as tif
        self.save_button_single = tk.Button(self.tab3, text="Export current frame", bg='grey', relief='groove', border=3,
                                            command=self.export_currently_displayed_image, font=self.fontsetting)

        # widget to save all images as tif
        self.save_button_all = tk.Button(self.tab3, text="Export all frames", bg='grey', border=3,
                                         command=self.set_image_export_all, font=self.fontsetting, relief='groove')

        # Section with advanced saving
        self.advanced_saving_label = tk.Label(self.tab3, text=f"Advanced Export", font=self.fontsetting, bg='grey')

        # widget to save image range as tif
        self.save_from_label = tk.Label(self.tab3, text=f"from:", font=self.fontsetting, bg='grey')
        self.save_from_value = tk.StringVar()
        self.save_from_entry = tk.Entry(self.tab3, textvariable=self.save_from_value, font=self.fontsetting, validate='key')
        self.save_from_entry['validatecommand'] = (self.save_from_entry.register(shf.testVal), '%P', '%d')
        self.save_from_entry.insert(0, "0")
        self.save_from = int(self.save_from_value.get())
        self.save_to_label = tk.Label(self.tab3, text=f"to:", font=self.fontsetting, bg='grey')
        self.save_to_value = tk.StringVar()
        self.save_to_entry = tk.Entry(self.tab3, textvariable=self.save_to_value, font=self.fontsetting, validate='key')
        self.save_to_entry['validatecommand'] = (self.save_to_entry.register(shf.testVal), '%P', '%d')
        self.save_to_entry.insert(0, f"{self.number_of_images - 1}")
        self.save_to = int(self.save_to_value.get())
        self.save_range_button = tk.Button(self.tab3, text="Export frame interval", bg='grey', font=self.fontsetting,
                                           command=self.set_image_export_range, relief='groove', border=3)

        # widget to export video
        self.export_video_button = tk.Button(self.tab3, text="Export video interval", bg='grey', border=3,
                                             command=self.export_video, font=self.fontsetting, relief='groove')
        self.video_fps_label = tk.Label(self.tab3, text=f"fps:", font=self.fontsetting, bg='grey')
        self.video_fps_value = tk.StringVar()
        self.video_fps_entry = tk.Entry(self.tab3, textvariable=self.video_fps_value, font=self.fontsetting, validate='key')
        self.video_fps_entry['validatecommand'] = (self.video_fps_entry.register(shf.testVal), '%P', '%d')
        self.video_fps_entry.insert(0, "30")

        # widget to export numpy binary
        self.export_np_binary_button = tk.Button(self.tab3, text="Export numpy binary interval", font=self.fontsetting,
                                                 bg='grey', command=self.export_np_binary, relief='groove', border=3)
        # a progressbar for export options
        self.video_export_status_label = tk.Label(self.tab3, text=f"Export status:", font=self.fontsetting, bg='grey')

        self.video_export_progressbar = ttk.Progressbar(self.tab3, mode='determinate', orient='horizontal')

        # Widget placement
        self.parent_directory_label.grid(row=0, column=0, columnspan=6, sticky='nsew', pady=2)
        self.parent_directory_entry.grid(row=1, column=0, columnspan=6, sticky='nsew', pady=2)

        self.simple_saving_label.grid(row=2, column=0, columnspan=6, sticky='nsew', pady=(20, 2))
        self.save_button_single.grid(row=3, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.save_button_all.grid(row=3, column=3, columnspan=3, sticky='nsew', padx=2, pady=2)

        self.advanced_saving_label.grid(row=4, column=0, columnspan=6, sticky='nsew', pady=2)
        self.save_from_label.grid(row=5, column=0, sticky='ew', padx=2, pady=2)
        self.save_from_entry.grid(row=5, column=1, sticky='ew', padx=2, pady=2)
        self.save_to_label.grid(row=5, column=2, sticky='ew', padx=2, pady=2)
        self.save_to_entry.grid(row=5, column=3, sticky='ew', padx=2, pady=2)
        self.save_range_button.grid(row=5, column=4, columnspan=2, sticky='nsew', padx=2, pady=2)

        self.export_video_button.grid(row=6, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.video_fps_label.grid(row=6, column=3, sticky='ew', padx=2, pady=2)
        self.video_fps_entry.grid(row=6, column=4, sticky='ew', padx=2, pady=2)
        self.export_np_binary_button.grid(row=7, column=0, columnspan=6, sticky='nsew', padx=2, pady=2)
        self.video_export_status_label.grid(row=8, column=0, columnspan=6, sticky='nsew', padx=2, pady=(20, 2))
        self.video_export_progressbar.grid(row=9, column=0, columnspan=6, sticky='nsew', padx=2, pady=2)


        ############################################## Tab 4 Widgets ###################################################
        ################################################################################################################
        ################################################################################################################

        self.load_files_label = tk.Label(self.tab4, text=f"Load files", font=self.fontsetting, bg='grey', height=2)

        # Data import settings widgets
        self.data_framerate_label = tk.Label(self.tab4, text=f"Frame rate [Hz]:", font=self.fontsetting, bg='grey')
        self.data_framerate_value = tk.StringVar()
        self.data_framerate_entry = tk.Entry(self.tab4, textvariable=self.data_framerate_value, font=self.fontsetting, validate='key')
        self.data_framerate_entry['validatecommand'] = (self.data_framerate_entry.register(shf.testVal), '%P', '%d')
        self.data_framerate_entry.insert(0, str(self.fps))
        self.data_framerate = int(self.data_framerate_value.get())
        self.data_pixelsize_label = tk.Label(self.tab4, text=f"Pixel size [µm]:", font=self.fontsetting, bg='grey')
        self.data_pixelsize_value = tk.StringVar()
        self.data_pixelsize_entry = tk.Entry(self.tab4, textvariable=self.data_pixelsize_value, font=self.fontsetting)
        self.data_pixelsize_entry.insert(0, str(self.pixel_size))
        self.data_pixelsize = float(self.data_pixelsize_value.get())

        # Load data raw widget
        self.load_data_button = tk.Button(self.tab4, text="Data:", bg='grey', command=self.select_data_array_path,
                                          font=self.fontsetting, relief='groove', border=3)
        self.load_data_value = tk.StringVar()
        self.load_data_entry = tk.Entry(self.tab4, textvariable=self.load_data_value, font=self.fontsetting)

        # Raw data import range widgets
        self.load_from_label = tk.Label(self.tab4, text=f"First frame:", font=self.fontsetting, bg='grey')
        self.load_from_value = tk.StringVar()
        self.load_from_entry = tk.Entry(self.tab4, textvariable=self.load_from_value, font=self.fontsetting, validate='key')
        self.load_from_entry['validatecommand'] = (self.load_from_entry.register(shf.testVal), '%P', '%d')
        self.load_from_entry.insert(0, "0")
        self.load_from = int(self.load_from_value.get())
        self.load_to_label = tk.Label(self.tab4, text=f"Last frame:", font=self.fontsetting, bg='grey')
        self.load_to_value = tk.StringVar()
        self.load_to_entry = tk.Entry(self.tab4, textvariable=self.load_to_value, font=self.fontsetting, validate='key')
        self.load_to_entry['validatecommand'] = (self.load_to_entry.register(shf.testVal), '%P', '%d')
        self.load_to_entry.insert(0, f"500")
        self.load_to = int(self.load_to_value.get())

        # Load flat widgets
        self.load_flat_button = tk.Button(self.tab4, text="Flat:", bg='grey', command=self.select_flat_array_path,
                                          font=self.fontsetting, relief='groove', border=3)
        self.load_flat_value = tk.StringVar()
        self.load_flat_entry = tk.Entry(self.tab4, textvariable=self.load_flat_value, font=self.fontsetting)

        # Load dark widgets
        self.load_dark_button = tk.Button(self.tab4, text="Dark:", bg='grey', command=self.select_dark_array_path,
                                          font=self.fontsetting, relief='groove', border=3)
        self.load_dark_value = tk.StringVar()
        self.load_dark_entry = tk.Entry(self.tab4, textvariable=self.load_dark_value, font=self.fontsetting)

        # Frame averaging number widgets
        self.flat_load_from_label = tk.Label(self.tab4, text=f"Frame avg. start:", font=self.fontsetting, bg='grey')
        self.flat_load_from_value = tk.StringVar()
        self.flat_load_from_entry = tk.Entry(self.tab4, textvariable=self.flat_load_from_value, font=self.fontsetting, validate='key')
        self.flat_load_from_entry['validatecommand'] = (self.flat_load_from_entry.register(shf.testVal), '%P', '%d')
        self.flat_load_from_entry.insert(0, "0")
        self.flat_load_from = int(self.load_from_value.get())
        self.flat_load_to_label = tk.Label(self.tab4, text=f"Frame avg. end:", font=self.fontsetting, bg='grey')
        self.flat_load_to_value = tk.StringVar()
        self.flat_load_to_entry = tk.Entry(self.tab4, textvariable=self.flat_load_to_value, font=self.fontsetting, validate='key')
        self.flat_load_to_entry['validatecommand'] = (self.flat_load_to_entry.register(shf.testVal), '%P', '%d')
        self.flat_load_to_entry.insert(0, f"50")
        self.flat_load_to = int(self.load_to_value.get())

        # Process files section label
        self.reprocess_newdata_label = tk.Label(self.tab4, text=f"File processing options:", font=self.fontsetting, bg='grey', height=2)

        # Import and crop data widgets
        self.import_rawdata_button = tk.Button(self.tab4, text="Import raw data", bg='grey', command=self.import_rawdata,
                                               font=self.fontsetting, relief='groove', border=3, height=2)
        self.import_flat_dark_button = tk.Button(self.tab4, text="Import flat/dark data", bg='grey', border=3, height=2,
                                                 command=self.import_flat_dark, font=self.fontsetting, relief='groove')

        # apply flat/dark correction widget
        self.apply_corrections_button = tk.Button(self.tab4, text="Apply flat/dark correction", bg='grey',
                                                  command=self.apply_flat_dark_correction, font=self.fontsetting,
                                                  relief='groove', border=3, height=2)

        # crop images widget
        self.crop_data_button = tk.Button(self.tab4, text="Crop data", bg='grey', command=self.apply_crop_data,
                                          font=self.fontsetting, relief='groove', border=3, height=2)

        # import status widgets
        self.data_import_status_label = tk.Label(self.tab4, text=f"Processing status:", font=self.fontsetting, bg='grey', height=2)
        self.data_import_progressbar = ttk.Progressbar(self.tab4, length=400, mode='determinate', orient='horizontal')

        # update GUI widget
        self.update_gui_button = tk.Button(self.tab4, text="Update GUI with new data!", bg='grey', border=3,
                                           command=self.update_gui, font=self.fontsetting, relief='groove', height=2)

        # Widget placement
        self.load_files_label.grid(row=0, column=0, columnspan=6, sticky='nsew', pady=2)
        self.data_framerate_label.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=2, pady=(2, 30))
        self.data_framerate_entry.grid(row=1, column=2, sticky='nsew', padx=2, pady=(0, 30))
        self.data_pixelsize_label.grid(row=1, column=3, columnspan=2, sticky='nsew', padx=2, pady=(2, 30))
        self.data_pixelsize_entry.grid(row=1, column=5, sticky='nsew', padx=2, pady=(0, 30))

        self.load_data_button.grid(row=2, column=0, sticky='nsew', padx=2, pady=2)
        self.load_data_entry.grid(row=2, column=1, columnspan=5, sticky='nsew', padx=2, pady=2)
        self.load_from_label.grid(row=3, column=0, columnspan=2, sticky='nsew', padx=2, pady=(2, 30))
        self.load_from_entry.grid(row=3, column=2, sticky='nsew', padx=2, pady=(2, 30))
        self.load_to_label.grid(row=3, column=3, columnspan=2, sticky='nsew', padx=2, pady=(2, 30))
        self.load_to_entry.grid(row=3, column=5, sticky='nsew', padx=2, pady=(0, 30))

        self.load_flat_button.grid(row=4, column=0, sticky='nsew', padx=2, pady=2)
        self.load_flat_entry.grid(row=4, column=1, columnspan=5, sticky='nsew', padx=2, pady=2)
        self.load_dark_button.grid(row=5, column=0, sticky='nsew', padx=2, pady=2)
        self.load_dark_entry.grid(row=5, column=1, columnspan=5, sticky='nsew', padx=2, pady=2)
        self.flat_load_from_label.grid(row=6, column=0, columnspan=2, sticky='nsew', padx=2, pady=(2, 30))
        self.flat_load_from_entry.grid(row=6, column=2, sticky='nsew', padx=2, pady=(2, 30))
        self.flat_load_to_label.grid(row=6, column=3, columnspan=2, sticky='nsew', padx=2, pady=(2, 30))
        self.flat_load_to_entry.grid(row=6, column=5, sticky='nsew', padx=2, pady=(2, 30))

        self.reprocess_newdata_label.grid(row=7, column=0, columnspan=6, sticky='nsew', pady=(2, 10))
        self.import_rawdata_button.grid(row=8, rowspan=2, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.import_flat_dark_button.grid(row=8, rowspan=2, column=3, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.apply_corrections_button.grid(row=10, rowspan=2, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.crop_data_button.grid(row=10, rowspan=2, column=3, columnspan=3, sticky='nsew', padx=2, pady=2)

        self.data_import_status_label.grid(row=12, column=0, columnspan=6, sticky='nsew', pady=(20, 2))
        self.data_import_progressbar.grid(row=13, column=0, columnspan=6, sticky='nsew', padx=2, pady=2)

        self.update_gui_button.grid(row=14, rowspan=2, column=0, columnspan=6, sticky='nsew', padx=2, pady=30)

        ################################################################################################################
        # # mute stdout print in Terminal
        text_trap = io.StringIO()
        sys.stdout = text_trap

        # initialise the program (has to be last command)
        self.start_mainloop()

    @staticmethod
    def on_enter(e):
        if isinstance(e.widget, tk.Button): e.widget['relief'] = 'raised'

    @staticmethod
    def on_leave(e):
        if isinstance(e.widget, tk.Button): e.widget['relief'] = 'groove'

    def image_data_canvas_compare(self):
        size_compare = min(self.canvas_width / self.image_width, self.canvas_height / self.image_height)
        auto_scaling = round(size_compare, 2) - 0.01
        return auto_scaling

    # GUI action functions

    def callback_left(self, e):
        self.canvas.delete('red')
        self.x_left = int((e.x - self.x_shift) / self.resize_value)
        self.y_top = int((e.y - self.y_shift) / self.resize_value)
        self.canvas.create_oval(e.x - 5, e.y - 5, e.x + 5, e.y + 5, fill="red", tags='red')
        self.coordinate_a['text'] = f"L-Click: A = ({self.x_left}, {self.y_top}); top-left"

        self.distance_ab_value = int(
            np.sqrt((self.x_right - self.x_left) ** 2 + (self.y_top - self.y_bot) ** 2) * self.pixel_size)
        self.distance_ab_label['text'] = f"AB distance = {self.distance_ab_value} µm"
        self.velocity_ab_value = int(self.distance_ab_value / (1000 * self.neighbour_order_value / self.fps))
        self.velocity_ab_label['text'] = f"AB velocity = {self.velocity_ab_value} mm/s"

    def callback_right(self, e):
        self.canvas.delete('green')
        self.x_right = int((e.x - self.x_shift) / self.resize_value)
        self.y_bot = int((e.y - self.y_shift) / self.resize_value)
        self.canvas.create_oval(e.x - 5, e.y - 5, e.x + 5, e.y + 5, fill="green", tags='green')
        self.coordinate_b['text'] = f"R-Click: B = ({self.x_right}, {self.y_bot}); bot-right"

        self.distance_ab_value = int(
            np.sqrt((self.x_right - self.x_left) ** 2 + (self.y_top - self.y_bot) ** 2) * self.pixel_size)
        self.distance_ab_label['text'] = f"AB distance = {self.distance_ab_value} µm"
        self.velocity_ab_value = int(self.distance_ab_value / (1000 * self.neighbour_order_value / self.fps))
        self.velocity_ab_label['text'] = f"AB velocity = {self.velocity_ab_value} mm/s"

    def callback_angle(self, e):
        if self.clicked is False:
            self.canvas.delete('yellow')
            self.angle_coordinates['x1'] = int((e.x - self.x_shift) / self.resize_value)
            self.angle_coordinates['y1'] = int((e.y - self.y_shift) / self.resize_value)
            self.canvas.create_oval(e.x - 5, e.y - 5, e.x + 5, e.y + 5, fill="yellow", tags='yellow')
            self.angle_measurement['text'] = f"First point set, press again."
            self.clicked = True
        elif self.clicked is True:
            self.angle_coordinates['x2'] = int((e.x - self.x_shift) / self.resize_value)
            self.angle_coordinates['y2'] = int((e.y - self.y_shift) / self.resize_value)
            tan_angle = (self.angle_coordinates['y1'] - self.angle_coordinates['y2']) / (
                        self.angle_coordinates['x1'] - self.angle_coordinates['x2'])
            self.canvas.create_oval(e.x - 5, e.y - 5, e.x + 5, e.y + 5, fill="yellow", tags='yellow')
            self.image_tilt = round(np.arctan(tan_angle) * 180 / np.pi, 3)
            self.angle_measurement['text'] = f"Image tilt = {self.image_tilt}°"
            self.clicked = False
            self.angle_coordinates = {}
        else:
            print("An error has occurred!")

    def callback_zoom_centre(self, e):
        self.canvas.delete('cyan')
        self.x_centre_zoom = int(e.x - self.x_shift)
        self.y_centre_zoom = int(e.y - self.y_shift)
        self.canvas.create_oval(e.x - 5, e.y - 5, e.x + 5, e.y + 5, fill="cyan", tags='cyan')
        self.zoom_centre_button[
            'text'] = f"Zoom centre:\n x = {self.x_centre_zoom}\n y = {self.y_centre_zoom}\n(Press to reset)"

    def resizing(self, event):
        if event.widget == self.root:
            if getattr(self, "_after_id", None):
                self.root.after_cancel(self._after_id)
            self.canvas_width = int(2 * self.root.winfo_width() / 3)
            self.canvas_height = int(5 * self.root.winfo_height() / 6)
            # determine correction factor for proper centering of image in canvas
            self.x_shift = (self.canvas_width - self.image_width) // 2
            self.y_shift = (self.canvas_height - self.image_height) // 2
            # calculate actual UI window dimensions based on data image size and UI scaling
            if self.auto_image_resize is True:
                self.resize_value = self.image_data_canvas_compare()
                self.resize_scale.set(self.resize_value)
            self.redraw_image()

    def photo_image(self, image: np.ndarray):
        image_min = np.min(image)
        image_max = np.max(image)
        image = shf.convert_image(image, 0, 255, "uint8", image_min=image_min, image_max=image_max)
        height, width = image.shape
        data = f'P5 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
        img = tk.PhotoImage(width=width, height=height, data=data, format='PPM')
        self.canvas.image = img

        self.canvas.create_image(self.x_shift, self.y_shift, anchor="nw", image=self.canvas.image, tags='image')
        self.time_label["text"] = f"t = {round(self.current_image_number * self.time_increment, 2)} ms"

        self.currently_displayed_image = image
        return self.canvas.image, width, height

    def redraw_image(self):
        self.canvas.delete('image')
        # self.update_image_selection_scale()
        image_to_display = self.recalculate_image()
        # recalculate displayed image size for coordinates
        self.x_shift = (self.canvas_width - image_to_display.shape[1]) // 2
        self.y_shift = (self.canvas_height - image_to_display.shape[0]) // 2
        img, width, height = self.photo_image(image_to_display)

    def recalculate_image(self):
        image_array = self.apply_pre_processing()
        # check for post-processing routines and apply if set
        if self.clipping_check.get() == 1:
            image_array = shf.clip_image(image_array, self.clip_min_value, self.clip_max_value, silent=True)
        if self.gaussian_check.get() == 1:
            image_array = filters.gaussian(image_array, preserve_range=True, sigma=self.gaussian_sigma_value).astype(
                image_array.dtype)
        if self.nlmeans_check.get() == 1:
            image_array = shf.convert_image(image_array, 0, 65535, "uint16")
            image_array = cv2.fastNlMeansDenoising(image_array,
                                                   templateWindowSize=self.nlmeans_template_window_size_value,
                                                   searchWindowSize=self.nlmeans_search_window_size_value,
                                                   h=[self.nlmeans_h_value],
                                                   normType=cv2.NORM_L1)
        if self.clahe_check.get() == 1:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit_value, tileGridSize=self.clahe_tile_grid_size_value)
            if self.nlmeans_check.get() != 1: image_array = shf.convert_image(image_array, 0, 65535, "uint16")
            image_array = clahe.apply(image_array)

        if self.resize_value!= 1.0:
            image_array = shf.resize_array(image_array, self.resize_value)
        if self.zoom_value != 1.0:
            image_array = shf.zoom_array(image_array, self.zoom_value, (self.x_centre_zoom, self.y_centre_zoom))
        return image_array

    def apply_pre_processing(self):
        mode = self.pre_processing_value
        if mode == 1:
            image_array = np.copy(self.displayed_array[:, :, self.current_image_number]) - np.copy(
                self.displayed_array[:, :, 0])
        elif mode == 2:
            image_array = (np.copy(self.displayed_array[:, :, self.current_image_number + self.neighbour_order_value]) -
                           np.copy(self.displayed_array[:, :, self.current_image_number]))
        elif mode == 3:
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.copy(self.displayed_array[:, :, self.current_image_number])
                b = np.copy(self.displayed_array[:, :, 0])
                image_array = np.where(b == 0, np.where(a == 0, 1, 1), a / b)
        elif mode == 4:
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.copy(self.displayed_array[:, :, self.current_image_number + self.neighbour_order_value])
                b = np.copy(self.displayed_array[:, :, self.current_image_number])
                image_array = np.where(b == 0, np.where(a == 0, 1, 1), a / b)
        elif mode == 5:
            image_array = nielsen_algorithm.nielsen_linear_comb(
                np.copy(self.displayed_array[:, :,
                        self.current_image_number:self.current_image_number + self.neighbour_order_value + 1]),
                0, self.neighbour_order_value, self.delta_scaler_value, self.ratio_scaler_value)
        else:
            image_array = np.copy(self.displayed_array[:, :, self.current_image_number])

        return image_array

    def update_image_selection_scale(self):
        mode = self.pre_processing_value
        if mode == 1:
            self.number_of_images = self.displayed_array.shape[2]
        elif mode == 2:
            self.number_of_images = self.displayed_array.shape[2] - self.neighbour_order_value
        elif mode == 3:
            self.number_of_images = self.displayed_array.shape[2]
        elif mode == 4:
            self.number_of_images = self.displayed_array.shape[2] - self.neighbour_order_value
        elif mode == 5:
            self.number_of_images = self.displayed_array.shape[2] - self.neighbour_order_value
        else:
            self.number_of_images = self.displayed_array.shape[2]
        self.current_image_scale.config(from_=0, to=self.number_of_images - 1)
        if self.save_to >= self.number_of_images - 1:
            self.save_to_value.set(f"{self.number_of_images - 1}")

    def bake_settings_to_array(self):
        baked_array = np.zeros([self.image_height, self.image_width, self.number_of_images], dtype='float32')
        for i in tqdm(range(self.number_of_images), ncols=75):
            self.current_image_number = i
            image_array = self.recalculate_image()
            baked_array[:, :, i] = image_array
        return baked_array

    def display_next_image(self, e):
        self.current_image_number = int(e)
        self.redraw_image()

    def previous_frame(self):
        if self.current_image_number == 0:
            self.current_image_number = self.number_of_images - 1
        else:
            self.current_image_number -= 1
        self.current_image_scale.set(self.current_image_number)
        self.redraw_image()

    def next_frame(self):
        if self.current_image_number == self.number_of_images - 1:
            self.current_image_number = 0
        else:
            self.current_image_number += 1
        self.current_image_scale.set(self.current_image_number)
        self.redraw_image()

    def start_play_video(self):
        if self.play_state is False:
            self.play_state = True
            self.play_pause_button["text"] = "PAUSE"
            self.play_pause_button['bg'] = 'orange'
            self.loop_play()
        else:
            self.play_state = False
            self.play_pause_button["text"] = "PLAY"
            self.play_pause_button['bg'] = 'green'

    def loop_play(self):
        if self.play_state is True:
            self.redraw_image()
            self.current_image_scale.set(self.current_image_number)
            try:
                image_skip_number = min(int(self.image_skip_value.get()), self.number_of_images - 1)
            except ValueError:
                image_skip_number = 1
            try:
                play_from_value = min(int(self.play_from_value.get()), self.number_of_images - 1)
            except ValueError:
                play_from_value = 0
            try:
                play_to_value = min(int(self.play_to_value.get()), self.number_of_images - 1)
                if play_to_value <= play_from_value: play_to_value = self.number_of_images - 1
            except ValueError:
                play_to_value = self.number_of_images - 1
            if ((self.current_image_number + image_skip_number) >= min(play_to_value - 1, self.number_of_images - 1)
                    or self.current_image_number < play_from_value):
                self.current_image_number = play_from_value
            else:
                self.current_image_number += image_skip_number
            self.root.after(1, self.loop_play)

    def reset_settings(self):
        self.canvas.delete('all')
        self.clipping_checkbutton.deselect()
        self.clip_min_scale.set(self.clip_min_default)
        self.clip_max_scale.set(self.clip_max_default)
        self.gaussian_checkbutton.deselect()
        self.gaussian_sigma_scale.set(self.gaussian_sigma_default)
        self.nlmeans_checkbutton.deselect()
        self.nlmeans_template_window_size_scale.set(self.nlmeans_template_window_size_default)
        self.nlmeans_search_window_size_scale.set(self.nlmeans_search_window_size_default)
        self.nlmeans_h_scale.set(self.nlmeans_h_slider_default)
        self.nlmeans_h_value = self.nlmeans_h_default
        self.clahe_checkbutton.deselect()
        self.clahe_clip_limit_scale.set(self.clahe_clip_limit_default)
        self.clahe_tile_grid_size_scale.set(self.clahe_tile_grid_size_default)
        self.resize_scale.set(1)
        self.reset_zoom_settings()

    def reset_zoom_settings(self):
        self.canvas.delete('all')
        self.zoom_scale.set(1)
        self.x_centre_zoom = self.currently_displayed_image.shape[1] // 2
        self.y_centre_zoom = self.currently_displayed_image.shape[0] // 2
        self.zoom_centre_button['text'] = f"Zoom centre:\n Shift + R-Click\n to select\n(Press to reset)"
        self.redraw_image()

    def adjust_clip_min_value(self, e):
        self.clip_min_value = float(e)
        if self.clipping_check.get() == 1: self.redraw_image()

    def adjust_clip_max_value(self, e):
        self.clip_max_value = float(e)
        if self.clipping_check.get() == 1: self.redraw_image()

    def adjust_gaussian_sigma_value(self, e):
        self.gaussian_sigma_value = float(e)
        if self.gaussian_check.get() == 1: self.redraw_image()

    def adjust_nlmeans_template_window_size_value(self, e):
        self.nlmeans_template_window_size_value = int(e)
        if self.nlmeans_check.get() == 1: self.redraw_image()

    def adjust_nlmeans_search_window_size_value(self, e):
        self.nlmeans_search_window_size_value = int(e)
        if self.nlmeans_check.get() == 1: self.redraw_image()

    def adjust_nlmeans_h_value(self, e):
        self.nlmeans_h_value = int(float(e) / 100 * 65535)
        if self.nlmeans_check.get() == 1: self.redraw_image()

    def adjust_clahe_clip_limit_value(self, e):
        self.clahe_clip_limit_value = float(e)
        if self.clahe_check.get() == 1: self.redraw_image()

    def adjust_tile_grid_size_value(self, e):
        self.clahe_tile_grid_size_value = (int(e), int(e))
        if self.clahe_check.get() == 1: self.redraw_image()

    def adjust_resize_level(self, e):
        self.resize_value = float(e)
        self.reset_zoom_settings()

    def adjust_zoom_level(self, e):
        self.zoom_value = float(e)
        self.redraw_image()

    def select_pre_processing_mode(self, e):
        self.pre_processing_value = int(e)
        self.update_image_selection_scale()
        if self.current_image_number >= self.number_of_images:
            self.current_image_number = self.current_image_number_default
            self.current_image_scale.set(self.current_image_number)
        self.redraw_image()

    def select_neighbour_order(self, e):
        self.neighbour_order_value = int(e)
        self.update_image_selection_scale()
        if self.current_image_number >= self.number_of_images:
            self.current_image_number = self.current_image_number_default
            self.current_image_scale.set(self.current_image_number)
        self.redraw_image()

    def adjust_delta_scaler_value(self, e):
        self.delta_scaler_value = float(e)
        if self.pre_processing_value == 5: self.redraw_image()

    def adjust_ratio_scaler_value(self, e):
        self.ratio_scaler_value = float(e)
        if self.pre_processing_value == 5: self.redraw_image()

    def reset_to_first(self):
        self.current_image_number = self.current_image_number_default
        self.current_image_scale.set(self.current_image_number)
        self.redraw_image()

    def start_mainloop(self):
        self.display_next_image(0)
        if self.flat is not None: self.load_flat_button['bg'] = 'green'
        if self.flat is not None:
            if not self.dark.any():
                self.load_dark_button['bg'] = 'orange'
            else:
                self.load_dark_button['bg'] = 'green'
        self.root.mainloop()

    def export_currently_displayed_image(self):
        self.source_path = asksaveasfilename(initialdir=self.parent_directory_value.get(),
                                             title='Select Directory to save or create Folder',
                                             defaultextension='.tif', confirmoverwrite=True)
        if len(self.source_path) == 0:
            pass
        else:
            imwrite(str(PurePath(self.source_path)), self.currently_displayed_image)

    def set_image_export_all(self):
        self.source_path = askdirectory(initialdir=self.parent_directory_value.get(),
                                        title='Select Directory to save or create Folder')
        if len(self.source_path) == 0:
            pass
        else:
            self.save_from = 0
            self.save_to = self.number_of_images - 1
            self.save_from_value.set(f"{self.save_from}")
            self.save_to_value.set(f"{self.save_to}")
            self.current_image_number = 0
            self.export_images()

    def verify_export_range(self):
        try:
            self.save_from = int(self.save_from_value.get())
            self.save_to = int(self.save_to_value.get())
            if not 0 <= self.save_from < self.save_to or not self.save_from <= self.save_to < self.number_of_images:
                raise ValueError
        except ValueError:
            self.save_from = 0
            self.save_to = self.number_of_images - 1

    def set_image_export_range(self):
        self.source_path = askdirectory(initialdir=self.parent_directory_value.get(),
                                        title='Select Directory to save or create Folder')
        if len(self.source_path) == 0:
            pass
        else:
            self.verify_export_range()
            self.current_image_number = self.save_from
            self.export_images()

    def export_images(self):
        last_image = self.current_image_number
        self.video_export_status_label['text'] = f"Saving images..."
        for index, i in enumerate(range(self.save_from, self.save_to + 1, 1)):
            self.video_export_progressbar['value'] = 100 * index / (self.save_to - self.save_from)
            self.root.update_idletasks()
            self.current_image_number = i
            image_array = self.recalculate_image()
            if image_array.dtype not in ["uint16", "uint8"]:
                image_array = shf.convert_image(image_array, 0, 65535, "uint16")
            imwrite(str(PurePath(self.source_path, str(self.current_image_number).zfill(4) + ".tif")), image_array)
        self.video_export_progressbar['value'] = 100
        self.video_export_status_label['text'] = f"Finished!"
        self.current_image_number = last_image

    def export_video(self):
        last_image = self.current_image_number
        self.source_path = asksaveasfilename(initialdir=self.parent_directory_value.get(),
                                             title='Select Directory to save video',
                                             defaultextension='.mp4', confirmoverwrite=True)
        if len(self.source_path) == 0:
            pass
        else:
            self.verify_export_range()
            frame_size = (self.currently_displayed_image.shape[1], self.currently_displayed_image.shape[0])
            try:
                frame_rate = int(self.video_fps_value.get())
            except ValueError:
                frame_rate = 30
            self.video_export_status_label['text'] = f"Creating Video..."
            output = cv2.VideoWriter(self.source_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size, False)
            for index, i in enumerate(range(self.save_from, self.save_to + 1, 1)):
                self.video_export_progressbar['value'] = 100 * index / (self.save_to - self.save_from)
                self.root.update_idletasks()
                self.current_image_number = i
                image_array = self.recalculate_image()
                image = shf.convert_image(image_array, 0, 255, "uint8")
                output.write(image)
            output.release()
            self.video_export_progressbar['value'] = 100
            self.video_export_status_label['text'] = f"Finished!"
            self.current_image_number = last_image

    def export_np_binary(self):
        self.source_path = asksaveasfilename(initialdir=self.parent_directory_value.get(),
                                             title='Select Directory to save binary',
                                             defaultextension='.npy', confirmoverwrite=True)
        if len(self.source_path) == 0:
            pass
        else:
            self.verify_export_range()
            self.video_export_status_label['text'] = f"Creating Binary..."
            baked_array = np.zeros([self.currently_displayed_image.shape[0], self.currently_displayed_image.shape[1],
                                    self.save_to - self.save_from], dtype='float32')
            for index, i in enumerate(range(self.save_from, self.save_to, 1)):
                self.video_export_progressbar['value'] = 100 * index / (self.save_to - self.save_from)
                self.root.update_idletasks()
                self.current_image_number = i
                image_array = self.recalculate_image()
                baked_array[:, :, index] = image_array
            np.save(self.source_path, baked_array)
            self.video_export_progressbar['value'] = 100
            self.video_export_status_label['text'] = f"Finished!"

    def select_data_array_path(self):
        data_path = askopenfilename(title='Select a data file to be loaded',
                                    filetypes=[('All', '*'), ('Cine files', '.cine'), ('Binary', '.npy')])
        self.load_data_value.set(data_path)

    def select_flat_array_path(self):
        data_path = askopenfilename(title='Select a flat field file to be loaded',
                                    filetypes=[('All', '*'), ('Cine files', '.cine'), ('Binary', '.npy')])
        self.load_flat_value.set(data_path)

    def select_dark_array_path(self):
        data_path = askopenfilename(title='Select a dark field file to be loaded',
                                    filetypes=[('All', '*'), ('Cine files', '.cine'), ('Binary', '.npy')])
        self.load_dark_value.set(data_path)

    def import_rawdata(self):
        data_path = self.load_data_value.get()
        if len(data_path) == 0:
            self.data_import_status_label['text'] = f"Raw image data not found..."
            pass
        else:
            try:
                self.load_from = int(self.load_from_value.get())
                self.load_to = int(self.load_to_value.get())
                number_of_images = self.load_to - self.load_from
                counter = [1]
                if os.path.isfile(data_path):
                    conversion_thread = shf.CustomThread(target=shf.load_from_file, args=(data_path, self.load_from, self.load_to,
                                                         'float32', counter, True))
                elif os.path.isdir(data_path):
                    conversion_thread = shf.CustomThread(target=shf.load_from_images, args=(data_path, self.load_from, self.load_to,
                                                         '.tif', 'float32', counter, True))
                conversion_thread.start()
                self.file_name = str(os.path.basename(data_path))
                self.data_import_status_label['text'] = f"Loading {self.file_name} ..."
                while conversion_thread.is_alive():
                    self.data_import_progressbar['value'] = 100 * len(counter) / number_of_images
                    self.root.update_idletasks()
                    time.sleep(0.2)
                self.image_array = conversion_thread.join()
                self.data_import_status_label['text'] = f"{self.file_name} loaded!"
            except ValueError:
                self.data_import_status_label['text'] = f"Entries for data image range invalid!"
                pass

    def import_flat_dark(self):
        flat_data_path = self.load_flat_value.get()
        dark_data_path = self.load_dark_value.get()
        try:
            start_frame = int(self.flat_load_from_value.get())
            end_frame = int(self.flat_load_to_value.get())
        except ValueError:
            start_frame = 0
            end_frame = 50
        if len(flat_data_path) == 0:
            self.data_import_status_label['text'] = f"Flat field data not found..."
            pass
        else:
            self.data_import_status_label['text'] = f"Importing flat field data..."
            self.flat = shf.calculate_series_average(flat_data_path, start_frame, end_frame,
                                                     os.path.splitext(flat_data_path)[1],
                                                     'float32', disable_tqdm=True)
            self.load_flat_button['bg'] = 'green'
            if len(dark_data_path) == 0:
                self.dark = np.zeros(self.flat.shape, dtype='float32')
                self.data_import_status_label['text'] = f"Only flat field data imported!"
                self.load_dark_button['bg'] = 'orange'
            else:
                self.data_import_status_label['text'] = f"Importing dark field data..."
                self.dark = shf.calculate_series_average(dark_data_path, start_frame, end_frame,
                                                         os.path.splitext(dark_data_path)[1],
                                                         'float32', disable_tqdm=True)
                self.data_import_status_label['text'] = f"Flat and dark field data imported!"
                self.load_dark_button['bg'] = 'green'

    def apply_crop_data(self):
        self.data_import_status_label['text'] = f"Cropping/rotating data array..."
        array_length = self.image_array.shape[2]
        if self.image_tilt != 0.0:
            for j in range(array_length):
                self.image_array[:, :, j] = shf.rotate_image(self.image_array[:, :, j], self.image_tilt)
                self.data_import_progressbar['value'] = 100 * j / array_length
                self.root.update_idletasks()
        self.image_array = self.image_array[self.y_top:self.y_bot, self.x_left:self.x_right, :]
        self.data_import_status_label['text'] = f"Cropping/rotating finished! Reload GUI to show!"

    def apply_flat_dark_correction(self):

        if self.flat is None or self.dark is None:
            self.data_import_status_label['text'] = f"Error: No flat/dark data loaded!"
            return
        height = self.image_array.shape[0]
        width = self.image_array.shape[1]

        if self.flat.shape[0] != height or self.flat.shape[1] != width:
            self.data_import_status_label['text'] = f"Error: Must perform flat/dark correction before cropping!"
            pass
        else:
            self.data_import_status_label['text'] = f"Performing flat/dark field correction..."
            with np.errstate(divide='ignore', invalid='ignore'):
                dark_corrected_flat = (self.flat - self.dark)
            for j in range(self.image_array.shape[2]):
                with np.errstate(divide='ignore', invalid='ignore'):
                    dark_corrected_image = self.image_array[:, :, j] - self.dark
                    corrected_image = np.where(dark_corrected_flat == 0, 1, dark_corrected_image / dark_corrected_flat)
                    self.image_array[:, :, j] = corrected_image
                    self.data_import_progressbar['value'] = 100 * j / self.image_array.shape[2]
                    self.root.update_idletasks()
            self.data_import_status_label['text'] = f"Flat/dark correction completed!"

    def update_gui(self):
        self.displayed_array = self.image_array
        self.current_image_number = self.current_image_number_default
        self.currently_displayed_image = None
        self.image_width = self.displayed_array[:, :, 0].shape[1]
        self.image_height = self.displayed_array[:, :, 0].shape[0]
        self.number_of_images = self.displayed_array.shape[2]

        self.root.title(
            f"SIA Viewer - File: {self.file_name}     (Image size: {int(self.image_width * self.pixel_size)} µm x {int(self.image_height * self.pixel_size)} µm)")
        try:
            self.fps = int(self.data_framerate_entry.get())
            self.pixel_size = float(self.data_pixelsize_entry.get())
        except ValueError:
            self.data_import_status_label['text'] = f"Invalid pixel or fps entry! Using old values."
        self.update_image_selection_scale()
        self.reset_to_first()
        self.resize_value = self.image_data_canvas_compare()
        self.resize_scale.set(self.resize_value)
        self.reset_zoom_settings()


"""Test the SIPS UI"""
# array = np.random.randint(255, size=(1500, 1500, 100), dtype='uint8')
# test = SIPS(array, 6600, 3.5)
