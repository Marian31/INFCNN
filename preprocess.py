import os
import glob
import cv2
import argparse
from helper.utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/test/clean_t0',
                    help='directory with test images')
parser.add_argument('--save_dir', dest='save_dir', default='./data/test',
                    help='directory path for saving results')
parser.add_argument('--prefix', dest='prefix', default='t', help='prefix for results folder name')
parser.add_argument('--image_format', dest='image_format', default='.png', help='format of images to process: jpg, png')
parser.add_argument('--noises', dest='noises', type=list, default=[0.1, 0.3, 0.4, 0.6], help='list of noises level')

args = parser.parse_args()


def gen_noisy_images(file_paths, clean_path, noise_path, nl, index=0):
    for i in range(len(file_paths)):
        image = cv2.imread(file_paths[i])
        noisy = get_noisy_image(image, nl)
        noisy = noisy.astype('uint8')
        cv2.imwrite(clean_path + "/%d00%d.png" % (index, i), noisy)
        cv2.imwrite(noise_path + "/%d00%d.png" % (index, i), image)


def main():
    save_dir = args.save_dir
    prefix = args.prefix

    print("[*] Reading test files...")
    file_paths_test = glob.glob(args.src_dir + '/*' + args.image_format)
    clean_path = f"{save_dir}/clean_{prefix}"
    noise_path = f"{save_dir}/noisy_{prefix}"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(clean_path):
        os.mkdir(clean_path)
    if not os.path.exists(noise_path):
        os.mkdir(noise_path)

    print("[*] Applying noise to test images...")
    for index, noise in enumerate(args.noises):
        gen_noisy_images(file_paths_test, clean_path, noise_path, noise, index)
    print("[*] Noisy and original images saved")


if __name__ == '__main__':
    main()
