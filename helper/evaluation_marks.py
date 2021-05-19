import tensorflow.compat.v1 as tf
from skimage.metrics import structural_similarity as ssim
import numpy as np

def cal_psnr(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def cal_ssim(im1, im2):
    return ssim(im1, im2, multichannel=True)


def show_average(psnr_sum, ssim_sum, size):
    avg_psnr = psnr_sum / size
    avg_ssim = ssim_sum / size
    print("---- Average PSNR %.2f ---" % avg_psnr)
    print("---- Average SSIM %.5f ---" % avg_ssim)