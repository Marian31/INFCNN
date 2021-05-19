import random
import numpy as np


def data_transformation(image, mode):
    if mode == 0:
        return image  # original
    elif mode == 1:
        return np.rot90(image)  # rotate 90 degree
    elif mode == 2:
        return np.rot90(image, k=2)  # rotate 180 degree
    elif mode == 3:
        return np.rot90(image, k=3)  # rotate 270 degree
    elif mode == 4:
        return np.flipud(image)  # flip up and down
    elif mode == 5:
        return np.flipud(np.rot90(image))   # rotate 90 degree and flip up and down
    elif mode == 6:
        return np.flipud(np.rot90(image, k=2))  # rotate 180 degree and flip
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))   # rotate 270 degree and flip


def get_random_int(max_value=255):
    return random.randint(0, max_value)


def get_noisy_image(image, nl):
    image_h, image_w, channels = image.shape
    number_of_corrupted_pixels = round(image_h * image_w * nl)
    map_of_corrupted_pixels = np.zeros(shape=(image_h, image_w))
    noisy_image = image.copy()
    for i in range(0, number_of_corrupted_pixels):
        while 1:
            x = get_random_int(image_h - 1)
            y = get_random_int(image_w - 1)
            if map_of_corrupted_pixels[x, y] == 0:
                break
        for k in range(channels):
            noisy_image[x, y, k] = get_random_int()  # if RGB channels, change R, G and B part of pixel to random
        map_of_corrupted_pixels[x, y] = 1
    return noisy_image
