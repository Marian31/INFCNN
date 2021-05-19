import argparse
import glob
import cv2
import os
from helper.utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/', help='directory of data')
parser.add_argument('--database', dest='database', default='coco2017', help='database with images')
parser.add_argument('--image_format', dest='image_format', default='jpg', help='format of images to process: jpg, png')
parser.add_argument('--patch_dir', dest='patch_dir', default='./data', help='dir of patches')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=39, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=39, help='stride')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
parser.add_argument('--results_clean', dest='results_clean', default="./img_clean_patches",
                    help='file with clean patches')
parser.add_argument('--results_noisy', dest='results_noisy', default="./img_noisy_patches",
                    help='file with noisy patches')
parser.add_argument('--nl', dest='nl', type=float, default=0.6, help='noise level intensity in the training')
parser.add_argument('--scales', dest='scales', type=list, default=[1, 0.9, 0.8, 0.7],
                    help='list of scales of images for patches')
parser.add_argument('--noises', dest='noises', type=list, default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6],
                    help='list of noises level')
parser.add_argument('--transform_times', dest='transform_times', type=int, default=1,
                    help='number of image transformation(rotation, flip) for each image')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')

args = parser.parse_args()


def count_patches_in_dimension(image_dimension_length):
    return int((image_dimension_length - args.step) / args.stride)


def get_number_of_patches(file_paths, scales_array):
    count = 0
    for image_path in file_paths:
        image = cv2.imread(image_path)
        image_h, image_w, channels = image.shape
        for scale in scales_array:
            scaled_image_h = int(image_h * scale)
            scaled_image_w = int(image_w * scale)
            ### Test1: ###
            # scaled_image_w += int(np.floor(scaled_image_w / 2))
            count += count_patches_in_dimension(scaled_image_h) * count_patches_in_dimension(scaled_image_w)
    return count


def generate_image_patches():
    batch_size = args.batch_size
    patch_size = args.patch_size
    patch_dir = args.patch_dir
    database = args.database
    scales = args.scales
    stride = args.stride
    noises = args.noises
    step = args.step
    nl = args.nl
    transform_times = args.transform_times

    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)

    database_output = database + "_" + str(args.patch_size)
    clean_output = os.path.join(patch_dir, args.results_clean + '_' + database_output + '.npy')
    noisy_output = os.path.join(patch_dir, args.results_noisy + '_' + database_output + '.npy')
    if os.path.isfile(clean_output) and os.path.isfile(noisy_output):
        print("The patches are generated.")
        return

    database_path = args.src_dir + '/' + database + '/*.' + args.image_format
    image_paths = glob.glob(database_path)
    print(f"Number of training images: {len(image_paths)}")

    # calculating the number of patches for initiating array
    count_patches_number = get_number_of_patches(image_paths, scales)
    num_patches = count_patches_number * transform_times

    if num_patches % args.batch_size:
        num_patches = (int(num_patches / batch_size) + 1) * batch_size
    print(f"Number of patches: {num_patches}; batch size: {batch_size}; number of batches: {num_patches / batch_size}")

    # initialize 4-D matrix of clear and noisy patches
    clean_output_data = np.zeros((num_patches, patch_size, patch_size, 3), dtype="uint8")
    noisy_output_data = np.zeros((num_patches, patch_size, patch_size, 3), dtype="uint8")

    # generate patches
    count = 0
    for i, image_path in enumerate(image_paths):
        print(f"Image: {i}")
        image = cv2.imread(image_path)
        image_h, image_w, channels = image.shape
        for scale in scales:
            scaled_image_h = int(image_h * scale)
            scaled_image_w = int(image_w * scale)
            resized_image = cv2.resize(image, (scaled_image_w, scaled_image_h), interpolation=cv2.INTER_CUBIC)
            for t in range(transform_times):
                for x in range(step, scaled_image_h - patch_size, stride):
                    for y in range(step, scaled_image_w - patch_size, stride):
                        patch = resized_image[x:x + patch_size, y:y + patch_size, :]
                        clean_array = data_transformation(patch, get_random_int(6))
                        noise_level = noises[get_random_int(len(noises) - 1)] if nl else nl
                        noisy_patch = get_noisy_image(clean_array, noise_level)
                        clean_output_data[count, :, :, :] = clean_array
                        noisy_output_data[count, :, :, :] = np.clip(noisy_patch, 0, 255).astype('uint8')
                        ### Test1: ###
                        # if y % 2:
                        #     count += 1
                        #     noisy = get_noisy_image(clean_array, noise_array[get_random_int(len(noise_array) - 1)])
                        #     clean_output_data[count, :, :, :] = clean_array
                        #     noisy_output_data[count, :, :, :] = np.clip(noisy, 0, 255).astype('uint8')
                        count += 1
    # pad the batch
    if count < num_patches:
        to_pad = num_patches - count
        clean_output_data[-to_pad:, :, :, :] = clean_output_data[:to_pad, :, :, :]
        noisy_output_data[-to_pad:, :, :, :] = noisy_output_data[:to_pad, :, :, :]

    np.save(clean_output, clean_output_data)
    np.save(noisy_output, noisy_output_data)
    print(f"Tensor size: {str(clean_output_data.shape)}")


if __name__ == '__main__':
    generate_image_patches()
