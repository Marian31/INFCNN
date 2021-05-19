import time
from helper.model_utils import *
from helper.evaluation_marks import *


def infcnn(input_data, is_training=True):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input_data, 64, 3, padding='same', activation=tf.nn.relu)
    for layer in range(2, 17):
        with tf.variable_scope('block%d' % layer):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layer, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output_mask = tf.layers.conv2d(output, 1, 3, padding='same', activation=tf.nn.sigmoid)
    return output_mask


class ImpulseNoiseFiltration:
    def __init__(self, sess, input_c_dim=3, nl=0.25, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.nl = nl
        self.batch_size = batch_size
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='clean_image')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='noisy_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.diff = tf.reduce_sum(tf.abs(self.X - self.Y_), 3) > 0.0
        self.diff_mask = tf.expand_dims(self.diff, 3)
        self.max_Y = tf.reduce_max(self.Y_)
        self.impulse_mask = infcnn(self.X, is_training=self.is_training)

        self.threshold = tf.placeholder(tf.float32, shape=(), name="threshold")
        self.mask = self.impulse_mask > self.threshold
        negative_mask = tf.logical_not(self.mask)
        impulse_mask3d = tf.concat([negative_mask, negative_mask, negative_mask], 3)

        self.Y = tf.where(impulse_mask3d, self.X, tf.zeros_like(self.X))
        self.loss = (1 / self.batch_size) * tf.nn.l2_loss(tf.to_float(self.diff_mask) - tf.to_float(self.impulse_mask))
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            graph = tf.Graph()
            writer = tf.summary.FileWriter('./graph', graph)
            writer.close()
            return True, global_step
        else:
            return False, 0

    def evaluate(self, iter_num, val_data_clean, val_data_noisy, sample_dir, summary_merged, summary_writer):
        print("[*] Evaluating...")
        psnr_sum, ssim_sum, loss_sum = 0, 0, 0
        for idx in range(len(val_data_clean)):
            clean_image = val_data_clean[idx].astype(np.float32) / 255.0
            clean_image_noisy = val_data_noisy[idx].astype(np.float32) / 255.0
            output_clean_image, noisy_image, org, impulse_mask, mask, loss, psnr_summary = self.sess.run(
                [self.Y, self.X, self.Y_, self.impulse_mask, self.mask, self.loss, self.eva_psnr],
                feed_dict={self.Y_: clean_image, self.X: clean_image_noisy, self.is_training: False,
                           self.threshold: 0.5})
            groundtruth = np.clip(val_data_clean[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            mask = impulse_mask < 0.5
            output_clean_image = noisy_image * mask
            reconstructed_img = mean_reconstruction(np.squeeze(output_clean_image), np.logical_not(np.squeeze(mask)), 1)
            reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype('uint8')
            loss_sum += loss

            # calculate PSNR and SSIM
            psnr = cal_psnr(groundtruth[0], reconstructed_img)
            im_ssim = cal_ssim(groundtruth[0], reconstructed_img)
            print(psnr, im_ssim, loss)
            print("img%d PSNR: %.2f;\tSSIM: %.5f\tLoss: %.4f" % (idx + 1, psnr, im_ssim, loss))
            psnr_sum += psnr
            ssim_sum += im_ssim
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)), groundtruth, noisyimage,
                        outputimage)
            save_images(os.path.join(sample_dir, 'original_mask%d_%d.png' % (idx + 1, iter_num)),
                        np.clip(255 * mask, 0, 255).astype('uint8'))
            save_images(os.path.join(sample_dir, 'mask%d_%d.png' % (idx + 1, iter_num)),
                        np.clip(255 * impulse_mask, 0, 255).astype('uint8'))

        show_average(psnr_sum, ssim_sum, len(val_data_clean))
        ev_loss = loss_sum / len(val_data_clean)
        print("--- Test ---- Evaluation_loss: %.5f ---" % ev_loss)

    def train(self, data_clean, data_noisy, eval_data_clean, eval_data_noisy, batch_size, ckpt_dir, epoches, lr,
              sample_dir, logs_dir, eval_every_epoch=1):
        # assert data range is between 0 and 1
        batches_number = int(data_clean.shape[0] / batch_size)
        max_iter_number = 60000
        max_steps = 1200
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // batches_number
            start_step = global_step % batches_number
            print("[*] Model restore success!")
        else:
            iter_num, start_epoch, start_step = 0, 0, 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        tf.summary.image('images', self.Y, 10)

        writer = tf.summary.FileWriter(logs_dir, self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data_clean, eval_data_noisy, sample_dir=sample_dir,
                      summary_merged=summary_psnr, summary_writer=writer)
        for epoch in range(start_epoch, epoches):
            p = np.random.permutation(data_clean.shape[0])
            data_clean = data_clean[p, :, :, :]
            data_noisy = data_noisy[p, :, :, :]
            steps, loss_sum, temp_step = 0, 0, 0
            for batch_id in range(start_step, batches_number):
                temp_step += 1
                if steps >= max_steps or iter_num >= max_iter_number:
                    break
                batch_images_clean = data_clean[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_images_noisy = data_noisy[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                _, loss, Y, max_Y, summary, XX, YY, impulse_mask, diff_mask = self.sess.run(
                    [self.train_op, self.loss, self.Y, self.max_Y, merged, self.X, self.Y_, self.impulse_mask,
                     self.diff_mask],
                    feed_dict={self.Y_: batch_images_clean, self.X: batch_images_noisy, self.lr: lr[epoch],
                               self.is_training: True, self.threshold: 0.5})
                loss_sum += loss
                if (batch_id + 1) % 100 == 0 or batch_id == 0 or batch_id == batches_number - 1:
                    print("Epoch: [%2d/%3d] [%4d/%4d] time: %4.4f, loss: %.6f"
                          % (epoch + 1, epoches, batch_id + 1, batches_number, time.time() - start_time,
                             loss_sum / temp_step))
                    temp_step, loss_sum = 0, 0
                iter_num += 1
                writer.add_summary(summary, iter_num)
                steps += 1

            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data_clean, eval_data_noisy, sample_dir=sample_dir,
                              summary_merged=summary_psnr, summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def test(self, test_files_clean, test_files_noisy, ckpt_dir, save_dir, _threshold=0.5):
        """Test INFCNN"""
        # init variables
        tf.global_variables_initializer()
        assert len(test_files_clean) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")

        psnr_sum, ssim_sum = 0, 0
        print("[*] " + 'noise level: ' + str(self.nl) + " start testing...")
        for idx in range(len(test_files_clean)):
            filename = os.path.basename(test_files_noisy[idx])
            if os.path.isfile(os.path.join(save_dir, 'denoised_CNN_' + filename)):
                continue
            clean_image = load_images(test_files_clean[idx]).astype(np.float32) / 255.0
            noisy_image = load_images(test_files_noisy[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image, impulse_mask = self.sess.run([self.Y, self.X, self.impulse_mask],
                                                                          feed_dict={self.Y_: clean_image,
                                                                                     self.X: noisy_image,
                                                                                     self.is_training: False,
                                                                                     self.threshold: _threshold})

            mask = impulse_mask < _threshold
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            output_clean_image = noisy_image * mask
            reconstructed_img = mean_reconstruction(np.squeeze(output_clean_image), np.logical_not(np.squeeze(mask)), 1)
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype('uint8')
            # calculate PSNR and SSIM
            psnr = cal_psnr(groundtruth, reconstructed_img)
            im_ssim = cal_ssim(groundtruth[0], reconstructed_img)
            print("img%d PSNR: %.2f;\tSSIM: %.5f" % (idx + 1, psnr, im_ssim))
            psnr_sum += psnr
            ssim_sum += im_ssim
            save_images(save_dir + '/detected_impulses_CNN_' + filename, outputimage)
            save_images(save_dir + '/denoised_CNN_' + filename, reconstructed_img)
        show_average(psnr_sum, ssim_sum, len(test_files_clean))

    def inference(self, test_image_name, ckpt_dir, save_dir, _threshold=0.5):
        """Model Test"""
        # init variables
        tf.global_variables_initializer()
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")

        y = np.empty([0, 0, 0, 3], dtype=float, order='C')
        noisy_image = load_images(test_image_name).astype(np.float32) / 255.0
        noisy_image, impulse_mask = self.sess.run([self.X, self.impulse_mask],
                                                  feed_dict={self.Y_: y,
                                                             self.X: noisy_image,
                                                             self.is_training: False,
                                                             self.threshold: _threshold})

        filename = os.path.basename(test_image_name)
        mask = impulse_mask < _threshold
        output_clean_image = noisy_image * mask

        reconstructed_img = mean_reconstruction(np.squeeze(output_clean_image), np.logical_not(np.squeeze(mask)), 1)
        outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
        reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype('uint8')

        save_images(os.path.join(save_dir, 'detected_impulses_CNN_' + filename), outputimage)
        save_images(os.path.join(save_dir, 'denoised_CNN_' + filename), reconstructed_img)

    def save(self, iter_num, checkpoint_dir, model_name='INFCNN'):
        saver = tf.train.Saver()
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=iter_num)
