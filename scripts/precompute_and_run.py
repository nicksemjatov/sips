import numpy as np
from tqdm import tqdm
from source import series_handling_functions as shf
from gui.sips_gui import SIPS
from pathlib import Path

"""An example script that shows how data can be loaded into SIPS via Python. This also allows to import any data type,
as long as it can be converted into 3 dimensional numpy stacks (x, y, z) and provided data is greyscale."""


flat_start_image = 3570  # start image of flat field recording
flat_end_image = 3580  # end frame of flat field recording
start_image = 150  # start image number for image analysis
end_image = 800  # end image number for image analysis
frame_rate = 25000  # frame rate of radiography data (necessary for TKViewer to determine image time)
pixel_size = 2.7  # pixel size of radiography data (necessary for TKViewer to determine image size)

# load data from folder structure
parent_directory: str = r"/mnt/data/Arbeitsdokumente/TEMP/point melt desy/"
files = ["536", "549", "552"]
data_path = str(Path(parent_directory, "sample", files[0] + ".cine"))
flat_field_path = str(Path(parent_directory, "flat", files[1] + ".cine"))
dark_field_path = str(Path(parent_directory, "dark", files[2] + ".cine"))

# load data manually
# data_path: str = r"/mnt/data/581.cine"
# flat_field_path: str = r"/mnt/data/592.cine"
# dark_field_path: str = r"D:\2023-07 DESY P61A Radiographiedaten\MM CMSX4 AZ3BV2\dark\2252.cine"

image_stack = shf.load_from_file(data_path, start_image=start_image, end_image=end_image, dtype='float32')
flat_field_average = shf.calculate_series_average(flat_field_path, flat_start_image, flat_end_image, file_type=".cine", dtype='float32')
dark_field_average = shf.calculate_series_average(dark_field_path, flat_start_image, flat_end_image, file_type=".cine", dtype='float32')
# create empty dark field if none is available
# dark_field_average = np.zeros(flat_field_average.shape, dtype='float32')

# perform flat field correction
with np.errstate(divide='ignore', invalid='ignore'):
    corrected_flat_field = (flat_field_average - dark_field_average)
for j in tqdm(range(image_stack.shape[2]), ncols=75):
    with np.errstate(divide='ignore', invalid='ignore'):
        a = image_stack[:, :, j] - dark_field_average
        b = corrected_flat_field
        corrected_image = np.where(b == 0, 1, a / b)
    image_stack[:, :, j] = corrected_image

# crop data to FOV before loading UI
#rotation_angle = 0
#x_left = 106
#x_right = 833
#y_top = 21
#y_bot = 534
# rotate and crop the image to extract relevant area
#if rotation_angle != 0.0:
#    for j in tqdm(range(image_stack.shape[2]),ncols=75):
 #       image_stack[:, :, j] = shf.rotate_image(image_stack[:, :, j], rotation_angle)
#image_stack = image_stack[y_top:y_bot, x_left:x_right, :]

measurement = SIPS(image_stack, frame_rate, pixel_size, flat=flat_field_average, dark=dark_field_average)

# bake pre-processing settings into image array for further processing
# image_stack = measurement.bake_settings_to_array()
# measurement = SIPS(image_stack, frame_rate, pixel_size, flat=flat_field_average, dark=dark_field_average)