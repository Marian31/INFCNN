import os
import argparse
import numpy as np
from glob import glob
import tensorflow.compat.v1 as tf
from helper.model import ImpulseNoiseFiltration
from helper.model_utils import load_data, load_images

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--nl', dest='nl', type=float, default=0.1, help='impulsive noise level')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint_impulses',
                    help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='sample_impulses', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test_impulses', help='test sample are saved here')
parser.add_argument('--eval_clean_set', dest='eval_clean_set', default='clean_v30',
                    help='dataset for eval in training')
parser.add_argument('--eval_noisy_set', dest='eval_noisy_set', default='noisy_v30',
                    help='dataset for eval in training')
parser.add_argument('--test_set_clean', dest='test_set_clean', default='clean_t30', help='dataset for testing')
parser.add_argument('--test_set_noisy', dest='test_set_noisy', default='noisy_t30', help='dataset for testing')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--database', dest='database', default='coco2017', help='database with images')
parser.add_argument('--results_clean', dest='results_clean', default="./data/img_clean_patches",
                    help='get pic from file')
parser.add_argument('--results_noisy', dest='results_noisy', default="./data/img_noisy_patches",
                    help='get pic from file')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=39, help='patch size')

args = parser.parse_args()

nl = args.nl
epoch = args.epoch
database_output = args.database + "_" + str(args.patch_size)
results_output = 'results/'
logs_dir = results_output + 'logs_' + database_output

if args.checkpoint_dir:
    checkpoint_dir = results_output + args.checkpoint_dir + '_' + database_output
if args.sample_dir:
    sample_dir = results_output + args.sample_dir + '_' + database_output
if args.test_dir:
    test_dir = results_output + args.test_dir + '_' + database_output


def filtration_train(filtration):
    clean_patches_path = args.results_clean + '_' + database_output + '.npy'
    noisy_patches_path = args.results_noisy + '_' + database_output + '.npy'
    print(f"Clean patches path: {clean_patches_path}")
    print(f"Noise patches path: {noisy_patches_path}")
    with load_data(filepath=clean_patches_path) as data_clean:
        with load_data(filepath=noisy_patches_path) as data_noisy:
            data_clean = data_clean.astype(np.float32) / 255.0  # normalize the data to 0-1
            data_noisy = data_noisy.astype(np.float32) / 255.0  # normalize the data to 0-1
            eval_noisy_files = sorted(glob(f"./data/test/{args.eval_noisy_set}/*.png"))
            eval_clean_files = sorted(glob(f"./data/test/{args.eval_clean_set}/*.png"))
            eval_noisy_images = load_images(eval_noisy_files)  # list of  images array of different size, 4-D
            eval_clean_images = load_images(eval_clean_files)  # list of images array of different size, 4-D
            print("Working...")

            lr = args.lr * np.ones([epoch])
            lr_adopted = [4, 10, 20, 50, 100]
            for i in range(int((epoch - 1) / 5)):
                lr[int(5 * (i + 1)):] = lr[0] / lr_adopted[i]
            filtration.train(data_clean, data_noisy, eval_clean_images, eval_noisy_images, batch_size=args.batch_size,
                             ckpt_dir=checkpoint_dir, epoches=epoch, lr=lr,
                             sample_dir=sample_dir, logs_dir=logs_dir)


def filtration_test(filtration):
    int_nl = args.test_set_clean.split('_')[-1][-2:]
    test_dir_nl = test_dir + '_' + int_nl
    if not os.path.exists(test_dir_nl):
        os.mkdir(test_dir_nl)
    test_files_clean = sorted(glob('./data/test/{}/*.png'.format(args.test_set_clean)))
    test_files_noisy = sorted(glob('./data/test/{}/*.png'.format(args.test_set_noisy)))
    filtration.test(test_files_clean, test_files_noisy, ckpt_dir=checkpoint_dir, save_dir=test_dir_nl)


def main(_):
    if not os.path.exists(results_output):
        os.mkdir(results_output)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # added gpu memory control
    print("GPU\n" if args.use_gpu else "CPU\n")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9) if args.use_gpu else None
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = ImpulseNoiseFiltration(sess, nl=nl)
        if args.phase == 'train':
            filtration_train(model)
        elif args.phase == 'test':
            filtration_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)


if __name__ == '__main__':
    tf.app.run()
