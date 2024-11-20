import numpy as np
import source.series_handling_functions as shf


def sum_differences(image_stack, start_index, frame_count):
    image_shape = image_stack.shape[:-1]
    sum_image = np.zeros(image_shape)
    for i in range(start_index+1, start_index+frame_count+1, 1):
        difference = calculate_difference(image_stack[:, :, start_index], image_stack[:, :, start_index + i])
        sum_image += difference
    return sum_image


def sum_ratios(image_stack, start_index, frame_count, q):
    image_shape = image_stack.shape[:-1]
    sum_image = np.zeros(image_shape)
    for i in range(start_index+1, start_index+frame_count+1, 1):
        ratio = calculate_ratio(image_stack[:, :, start_index], image_stack[:, :, start_index + i], q)
        sum_image += ratio
    return sum_image


def calculate_difference(base_image, subtracted_image):
    difference = base_image - subtracted_image
    difference = nielsen_sat_function(difference)
    return difference


def calculate_ratio(base_image, divided_image, q):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(divided_image == 0, 1, base_image / divided_image)
    ratio = nielsen_sat_function(q*ratio)
    return ratio


def nielsen_sat_function(image_array):
    # crops the intensity range to standard 8-bit integers
    lowest = 0
    highest = 255
    sat_image = np.where(image_array < lowest, lowest, np.where(image_array > highest, highest, image_array))
    return sat_image


def nielsen_linear_comb(image_stack, start_index, frame_count, d, q):
    # if image_stack.dtype != "'uint8":
    #     image_stack = shf.convert_image(image_stack, 0, 255, 'uint8')
    linear_comb = d * sum_differences(image_stack, start_index, frame_count) + sum_ratios(image_stack, start_index, frame_count, q)
    return linear_comb
