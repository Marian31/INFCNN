import argparse
import tensorflow.compat.v1 as tf
from helper.model import ImpulseNoiseFiltration

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--test_file', dest='test_file', default='./data/test/noisy_t0/0000.png', help='test file name')
parser.add_argument('--database', dest='database', default='coco2017', help='database with images')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint_impulses',
                    help='models are saved here')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=39, help='patch size')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')

args = parser.parse_args()

if args.checkpoint_dir:
    checkpoint_dir = args.checkpoint_dir
if args.save_dir:
    results_dir = args.save_dir
else:
    results_dir = '.'

database_output = args.database + "_" + str(args.patch_size)
checkpoint_dir = 'results/' + args.checkpoint_dir + '_' + database_output


def filtration_inference(filtration):
    filtration.inference(args.test_file, ckpt_dir=checkpoint_dir, save_dir=results_dir)


def main(_):
    print("GPU\n" if args.use_gpu else "CPU\n")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9) if args.use_gpu else None
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = ImpulseNoiseFiltration(sess)
        filtration_inference(model)


if __name__ == '__main__':
    tf.app.run()
