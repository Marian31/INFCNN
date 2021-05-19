import gc
import os
import sys
from PIL import Image
import numpy as np


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        # np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath='./data/image_clean_pat.npy'):
    return DataLoader(filepath=filepath)


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('RGB')
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        im = Image.open(file).convert('RGB')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('RGB')
    im.save(filepath, 'png')


def mean_reconstruction(img=None, mask=None, threshold=None):
    img = np.asarray(img)
    mask = np.asarray(mask)
    img_out = np.copy(img)
    x = img.shape[0]
    y = img.shape[1]
    for i in range(0, x):
        for j in range(0, y):
            if (mask[i, j] > 0):
                param = 1
                while True:
                    xlow = max(i - param, 0)
                    ylow = max(j - param, 0)
                    xhigh = min(i + param + 1, x)
                    yhigh = min(j + param + 1, y)
                    submask = np.logical_not(mask[xlow:xhigh, ylow:yhigh])
                    if np.sum((submask)) >= threshold:
                        break
                    else:
                        param = param + 1
                xlow = max(i - param, 0)
                ylow = max(j - param, 0)
                xhigh = min(i + param + 1, x)
                yhigh = min(j + param + 1, y)
                pixels = np.empty((0, 3), int)
                for a in range(xlow, xhigh):
                    for b in range(ylow, yhigh):
                        if mask[a, b] == 0:
                            tmp = np.reshape(img[a, b, 0:3], (1, 3))
                            pixels = np.append(pixels, tmp, axis=0)
                tmp_out = np.mean(pixels, axis=0)
                img_out[i, j, :] = np.reshape(tmp_out, (1, 1, 3))
    return img_out  # Image.fromarray(imgOut, 'RGB')
