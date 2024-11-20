import cv2
import numpy as np
from pathlib import PurePath, Path
import os
import shutil
from tqdm import tqdm
from threading import Thread
from natsort import natsorted


class CustomThread(Thread):
    """
    A custom Thread class that allows to retrieve the return value from a function that was executed in a Thread.
    """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def testVal(inStr, acttyp):

    """
    A small function to validate whether tkinter entry input is a digit and block insertion if not
    :param inStr:
    :param acttyp:
    :return:
    """
    if acttyp == '1':  # insert
        if not inStr.isdigit():
            return False
    return True


def create_dir(path: str):

    """
    Create a directory. Delete the directory if it already exists.

    :param path:
           Path name of directory
    """
    exists = os.path.exists(path)
    if exists:
        eingabe = input('The file/folder already exists, '
                        'Do you really want to delete it and create a new folder?  y or n\n')
        if eingabe == 'y' or 'yes':
            shutil.rmtree(path)
            os.makedirs(path)
            pass
        else:
            print('the process is ending')
            exit()
    else:
        os.makedirs(path)


def calculate_series_average(path: str, start_image=0, end_image=100, file_type=".tif", dtype='uint16', disable_tqdm=False):

    """
    Function to calculate the average pixel values from a given stack of images (last axis denotes image number).
    """

    if file_type == ".cine":
        series_stack = load_from_file(path, start_image, end_image, dtype, disable_tqdm=disable_tqdm)
    elif file_type == ".npy":
        series_stack = np.load(path)
    elif file_type in [".tif", ".tiff"]:
        series_stack = load_from_images(path, start_image, end_image, file_type, dtype)
    else:
        print("File type not recognized")
        return 0

    mean_of_series = np.mean(series_stack, axis=2, dtype=dtype)
    print(f'Numpy array info: dtype = {mean_of_series.dtype}, shape = {mean_of_series.shape}, min = {np.min(mean_of_series)}, max = {np.max(mean_of_series)}')
    return mean_of_series


def load_from_images(directory, start_image=0, end_image=100, file_type=".tif", dtype='uint16', counter=[0], disable_tqdm=False):

    """
    Function to load images in a given folder and save them as a 3D numpy array (last axis denotes image number).
    """

    file_names = os.listdir(directory)
    file_names = [x for x in file_names if x.endswith(file_type)]
    file_names = natsorted(file_names)
    if end_image >= len(file_names): end_image = len(file_names)-1

    print(f"Loading frames from image files in {Path(directory).stem}...\n", end="")
    first_image = cv2.imread(str(PurePath(directory, file_names[start_image])), cv2.IMREAD_UNCHANGED)
    height = first_image.shape[0]
    width = first_image.shape[1]
    image_stack = np.zeros((height, width, end_image-start_image+1), dtype=dtype)
    for i in tqdm(range(end_image - start_image+1), ncols=75, disable=disable_tqdm):
        file_path = str(PurePath(directory, file_names[i+start_image]))
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        image_stack[:, :, i] = image
        counter.append(i)
    print("...loading completed!")
    print(f'Image stack info: dtype = {image_stack.dtype}, shape = {image_stack.shape}, min = {np.min(image_stack)}, max = {np.max(image_stack)}')
    return image_stack


def load_from_file(file_path, start_image=0, end_image=100, dtype='uint16', counter=[0], disable_tqdm=False):

    """
    Function to load images saved in .cine or .npy files into numpy arrays (last axis denotes image number).
    """

    if file_path.endswith('.npy'):
        print(f"Loading frames from {Path(file_path).name}...", end="")
        image_stack = np.load(file_path)
        print("loading completed!")
        return image_stack

    # load necessary packages to import from cine only when this function is called
    elif file_path.endswith('.cine'):
        try:
            from pycine.raw import read_frames
            from pycine.file import read_header
        except ModuleNotFoundError:
            print("You forgot to install the pycine package! Please refer to the Readme for installation instructions.")
            return

        file_header = read_header(file_path)
        width, height = file_header["bitmapinfoheader"].biWidth, file_header["bitmapinfoheader"].biHeight
        total_image_count = file_header["cinefileheader"].TotalImageCount
        if end_image > total_image_count - 1:
            end_image = total_image_count - 1
            print(f"Video has only {total_image_count} frames. Frame count starts at 0.")
        if start_image > end_image: start_image = end_image

        print(f"Loading frames from {Path(file_path).name}...\n", end="")
        image_stack = np.zeros((height, width, end_image-start_image+1), dtype=dtype)
        raw_images, setup, bpp = read_frames(file_path, start_frame=start_image, count=end_image-start_image+1)
        for i in tqdm(range(end_image-start_image+1), ncols=75, disable=disable_tqdm):

            counter.append(i)

            image_stack[:, :, i] = next(raw_images)
        print("...loading completed!")
        print(f'Image stack info: dtype = {image_stack.dtype}, shape = {image_stack.shape}, min = {np.min(image_stack)}, max = {np.max(image_stack)}')
        return image_stack
    else:
        print("File type not recognised. Use .cine or .npy!")
        return


def video_from_image(image_directory: str, output_path: str, image_limit=1000, file_type=".tif", frame_rate=30, color=False):

    """
    Create a video from images in a given folder.
    """

    file_names = os.listdir(image_directory)
    file_names = [x for x in file_names if x.endswith(file_type)]
    from natsort import natsorted
    file_names = natsorted(file_names)

    if len(file_names) < image_limit: image_limit = len(file_names)

    first_frame = cv2.imread(str(PurePath(image_directory, file_names[0])), cv2.IMREAD_UNCHANGED)

    array_shape = first_frame.shape
    print(array_shape)
    frame_size = (array_shape[1], array_shape[0])
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size, color)

    print("Making video from images...\n", end="")
    for i in tqdm(range(image_limit), ncols=75):
        image = cv2.imread(str(PurePath(image_directory, file_names[i])), cv2.IMREAD_UNCHANGED)
        output.write(image)
    output.release()
    print("...video created!")


def video_from_array(array: np.ndarray, output_path: str, frame_rate=30):

    """
    Create a video from images in a 3D numpy array (last axis denotes image number).
    """

    array_shape = array.shape
    frame_size = (array_shape[1], array_shape[0])
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size, False)

    print("Writing video...\n", end="")
    for i in tqdm(range(array.shape[2]), ncols=75):
        # call function to convert numpy array images from current data-type to uint8
        image = convert_image(array[:, :, i], 0, 255, "uint8")
        output.write(image)
    output.release()
    print("...video created!")


def rescale_video(input_path: str, output_path: str, fps=30, size=(1920, 1080), color=False):

    """
    Rescale the image size of a video (e.g. downscale from 4k to FullHD).
    """

    vidcap = cv2.VideoCapture(input_path)
    success, image = vidcap.read()
    i = 0

    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, color)

    while success:
        success, image = vidcap.read()
        resize = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        output.write(resize)
        i += 1
    vidcap.release()
    output.release()


def convert_image(image, target_type_min, target_type_max, target_type, image_min=None, image_max=None):

    """
    Convert image from one dtype to another. By default, min and max value of input image are scaled to target_type
    min and max values of output dtype (equivalent to preserve_range=False from scikit-image).
    """

    if image_min is None:
        image_min = image.min()
    if image_max is None:
        image_max = image.max()

    # escape in case of empty image array to avoid runtime error
    if image_min == image_max:
        return np.zeros(image.shape, dtype=target_type)

    with np.errstate(divide='ignore', invalid='ignore'):
        a = (target_type_max - target_type_min) / (image_max - image_min)
    b = target_type_max - a * image_max
    converted_image = (a * image + b).astype(target_type)

    return converted_image


def clip_image(image: np.ndarray, min_percentile: float, max_percentile: float, silent=False):

    """
    Function to clip images to remove edge outlier pixels and improve contrast + brightness.
    """

    clip_min, clip_max = np.percentile(image, [min_percentile, max_percentile])
    if silent is False:
        print(f"Before clipping, the image has a value range from {np.min(image)} to {np.max(image)}")

    if image.ndim == 2:
        image[image <= clip_min] = clip_min
        image[image >= clip_max] = clip_max
    elif image.ndim == 3:
        for j in tqdm(range(image.shape[2]), ncols=75):
            frame = image[:, :, j]
            frame[frame <= clip_min] = clip_min
            frame[frame >= clip_max] = clip_max
            image[:, :, j] = frame
    else:
        print("Your image has the wrong number of dimensions (has to be 2 or 3)!")
        print(image.ndim)
    if silent is False:
        print(f"After clipping the image has a value range from {np.min(image)} to {np.max(image)}.")
    return image


def resize_array(image_array: np.ndarray, resize_factor: float):

    """
    Resize a 2D numpy array to a smaller or larger size through value interpolation. Can be used to increase or
    decrease real image size.
    """

    target_width = int(image_array.shape[1] * resize_factor)
    target_height = int(image_array.shape[0] * resize_factor)
    resized_array = cv2.resize(image_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_array


def zoom_array(image_array: np.ndarray, zoom_factor: float, zoom_coord=None):

    """
    Zoom into an image array, with zoom centre defined by zoom_coord.
    """

    cy, cx = [i / 2 for i in image_array.shape[:2]] if zoom_coord is None else zoom_coord[::-1]
    rot_matrix = cv2.getRotationMatrix2D((cx, cy), 0, zoom_factor)
    zoomed_array = cv2.warpAffine(image_array, rot_matrix, image_array.shape[1::-1], flags=cv2.INTER_AREA)
    return zoomed_array


def rotate_image(image: np.ndarray, angle: float):

    """
    Rotate an image around its centre point counter-clockwise.
    """

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

