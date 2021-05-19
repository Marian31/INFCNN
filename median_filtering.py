import glob
import cv2
import os
import argparse
from helper.evaluation_marks import *
from helper.model_utils import save_images

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/test/', help='directory of data')
parser.add_argument('--clear', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--noisy', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--median_core', dest='median_core', type=int, default=7,
                    help='size of median filter kernel, only odd numbers: 3, 5, 7...')
parser.add_argument('--nl', dest='nl', type=float, default=0.1, help='impulsive noise level')
parser.add_argument('--save_dir', dest='save_dir', default='results/median', help='directory to save result')

args = parser.parse_args()

clear = 'clean_t0'
noisy = 'noisy_t0'
src_dir = args.src_dir
median_core = args.median_core
save_dir = args.save_dir + '_' + str(median_core)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def get_median_mark():
    clear_file_paths = glob.glob(src_dir + clear + '/*.png')
    noisy_file_paths = glob.glob(src_dir + noisy + '/*.png')
    psnr_sum = 0
    ssim_sum = 0
    for i in range(len(clear_file_paths)):
        clear_image = cv2.cvtColor(cv2.imread(clear_file_paths[i]), cv2.COLOR_BGR2RGB)
        noisy_image = cv2.cvtColor(cv2.imread(noisy_file_paths[i]), cv2.COLOR_BGR2RGB)
        median_image = cv2.medianBlur(noisy_image, median_core)
        psnr = cal_psnr(clear_image, median_image)
        im_ssim = cal_ssim(clear_image, median_image)
        print("img%d PSNR: %.2f;\tSSIM: %.5f" % (i + 1, psnr, im_ssim))
        psnr_sum += psnr
        ssim_sum += im_ssim
        save_images(os.path.join(save_dir, 'denoised_%000d.png' % i), median_image)
    show_average(psnr_sum, ssim_sum, len(clear_file_paths))


if __name__ == '__main__':
    get_median_mark()
