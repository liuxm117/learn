# _*_ coding: utf-8 _*_

import sys
sys.path.append('/mnt/data2/liuxin/')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import tensorflow as tf
import numpy as np
from datetime import datetime
import tensorflow.contrib.slim as slim
from AutoML_20181121.nasnet import nasnet
# from AutoML_20181121.encoder import encoder_utils
from AutoML_20181121.cifar10.data_utils import ImageDataGenerator1

batch_size = 128
num_class = 10
learning_rate = 0.001
decay_steps = 10000
decay_rate = 0.7
num_epochs = 200
data_path = '/mnt/data2/liuxin/AutoML_20181121/cifar10/datasets/'
layer_num = 20

train_display_step = 10
val_display_step = 10

generator = ImageDataGenerator1(data_path, n_class=10, batch_size=batch_size)
iterator = generator.iterator
X_test = generator.X_test
Y_test = generator.Y_test


def train_model():
    x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_class])

    # layer_coder = encoder_utils.get_net_random(layer_num, is_random=False)
    # cell_coder = encoder_utils.get_cell_random(is_random=False)
    cell_coder = np.array([[1,10,0,11,0],[1,11,1,10,0],[0,4,1,0,0],[1,4,1,4,0],[0,10,0,0,0],
                  [0,11,1,12,0], [0,5,1,12,0], [0,4,1,11,0], [3,0,2,4,0], [2,10,0,5,0]], dtype=np.int)
    # layer_coder = np.array([[0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0]], dtype=np.int)

    layer_coder = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.int)

    arg_scope = nasnet.nasnet_cifar_arg_scope(weight_decay=5e-4)
    with slim.arg_scope(arg_scope):
        logits, end_points = nasnet.build_nasnet_cifar(x, num_class, layer_coder, cell_coder, layer_num)
        if 'AuxLogits' in end_points:
            aux_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=end_points['AuxLogits'], labels=y))
            tf.add_to_collection('losses', aux_loss*0.4)
            tf.summary.scalar('aux_loss', aux_loss*0.4)
        else:
            aux_loss = tf.constant(0, dtype=tf.float32)
            tf.add_to_collection('losses', aux_loss * 0.4)

    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    tf.summary.scalar('losses', train_loss)
    tf.add_to_collection('losses', train_loss)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
    tf.add_to_collection('losses', regularization_loss*0.05)

    loss = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('total_loss', loss)

    global_s = tf.Variable(0, trainable=False)  #训练中是计数的作用，每训练一个batch就加1
    lr = tf.train.exponential_decay(learning_rate, global_s, decay_steps, decay_rate, staircase=True)
    lr = tf.maximum(lr, 5e-6)
    tf.summary.scalar('learning_rate', lr)

    # optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_s)
    # train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_s)

    # get all trainable variables in your model
    params = tf.trainable_variables()
    # create an optimizer
    opt = tf.train.AdamOptimizer(lr)
    # compute gradients for params
    gradients = tf.gradients(loss, params)
    # process gradients
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5)
    train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_s)

    correct_pred = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('train_accuracy', accuracy)

    # correct_pred_aux = tf.equal(tf.arg_max(end_points['AuxLogits'], 1), tf.arg_max(y, 1))
    # accuracy_aux = tf.reduce_mean(tf.cast(correct_pred_aux, tf.float32))
    # tf.summary.scalar('aux_accuracy', accuracy_aux)

    # 所有张量融合统一写入
    merged_summary = tf.summary.merge_all()
    # 写到指定的磁盘路径中
    train_writer = tf.summary.FileWriter('log/train')
    test_writer = tf.summary.FileWriter('log/test')
    # saver = tf.train.Saver(max_to_keep=2)

    train_batchs_per_epochs = np.floor(50000 / batch_size).astype(np.int16)
    test_batchs_per_epochs = np.floor(10000 / batch_size).astype(np.int16)

    log = open("test_results_v1.txt", "a+")

    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)
        pointer = 0
        test_var = []
        GL_total = []
        test_acc_total = []
        count = 0

        for epoch in range(num_epochs):
            print('Epochs number:{}'.format(epoch+1))
            step = 1

            while step < train_batchs_per_epochs:
                global_step = epoch * train_batchs_per_epochs + step

                # cifar10数据集测试
                # batch_xs, batch_ys = generator.getNext_batch_train()
                # batch_xs = X_train[pointer:pointer + batch_size]
                # batch_ys = Y_train[pointer:pointer + batch_size]
                # pointer += batch_size
                # sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})

                batch_trainX, batch_trainY = next(iterator)  # get the next batch
                if len(batch_trainX) < batch_size:
                    batch_trainX, batch_trainY = next(iterator)
                sess.run([train_op], feed_dict={x: batch_trainX, y: batch_trainY})

                if step % train_display_step == 0:
                    summary, logitsRe, lossRe, acc, regularization_loss1, aux_loss1, train_loss1, lr1 = sess.run([merged_summary, logits, loss, accuracy,
                                                               regularization_loss, aux_loss, train_loss, lr],
                                                               feed_dict={x: batch_trainX, y: batch_trainY})
                    train_writer.add_summary(summary, global_step)
                    log_str = "current time   " + str(datetime.now().time()) + "\n" \
                              "regularization_loss   " + str(regularization_loss1) + "\n" + \
                              "aux_loss   " + str(aux_loss1) + "\n" + \
                              "train_loss   " + str(train_loss1) + "\n" + \
                              "total_loss   " + str(lossRe) + "\n" + \
                              "train_acc   " + str(acc) + "\n" + \
                              "lr   " + str(lr1) + "\n"
                    print(log_str)
                    log.write(log_str)

                if step % val_display_step == 0:
                    if pointer + batch_size > 10000:
                        pointer = 0

                    batch_xs = X_test[pointer:pointer + batch_size]
                    batch_ys = Y_test[pointer:pointer + batch_size]
                    pointer += batch_size

                    summary, logitsRe, lossRe, acc = sess.run([merged_summary, logits, loss, accuracy],
                                                               feed_dict={x: batch_xs, y: batch_ys})
                    test_writer.add_summary(summary, global_step)
                    log_str = "val_loss   " + str(lossRe) + "\n" +\
                              "val_acc   " + str(acc) + "\n"
                    print(log_str)
                    log.write(log_str)
                step += 1
            # pointer = 0
            # generator.reset_pointer()
            # print('{} saving checkpoint of model ...'.format(datetime.now()))
            # checkpoint_name = os.path.join('log/', 'model_epoch' + str(epoch + 1) + '.ckpt')
            # saver.save(sess, checkpoint_name, write_meta_graph=False)
            # print('{} Model checkpoint saved at {}'.format(datetime.now(), checkpoint_name))

            total_acc = 0
            total_loss = 0
            pointer1 = 0
            for i in range(test_batchs_per_epochs):
                # cifar10数据集测试
                # batch_xs, batch_ys = generator.getNext_batch_test()
                batch_xs = X_test[pointer1:pointer1 + batch_size]
                batch_ys = Y_test[pointer1:pointer1 + batch_size]
                pointer1 += batch_size

                test_logits, test_loss, test_acc = sess.run([logits, loss, accuracy],
                                                             feed_dict={x: batch_xs, y: batch_ys})
                total_acc += test_acc
                total_loss += test_loss
            # pointer1 = 0
            acc = total_acc/test_batchs_per_epochs
            total_loss /= test_batchs_per_epochs
            log_str = "current time   " + str(datetime.now().time()) + "\n"  + \
                      "epoch   " + str(epoch) + "\n"  + \
                      "test loss  " + str(total_loss) + "\n"  + \
                      "test acc   " + str(acc) + "\n"
            print(log_str)
            log.write(log_str)

            test_acc_total.append(acc)
            test_var.append(total_loss)
            var_opt = min(test_var) + 1e-6
            var_va = test_var[-1] + 1e-6
            GL = 100 * (var_va / var_opt -1)
            GL_total.append(GL)

            # if test_var.index(min(test_var)) == len(test_var)-1:
            #     count = 0
            # elif GL_total[-1] > GL_total[-2]:
            #     count += 1
            # else:
            #     count = 0
            # if count >= 5:
            #     log_str = "stopping early" + "epoch " + str(epoch) + "\n"
            #     print(log_str)
            #     log.write(log_str)
            #     break

            if GL > 1e-4:
                count += 1
            else:
                count = 0
            if count >= 8:
                log_str = "stopping early" + "epoch " + str(epoch) + "\n"
                print(log_str)
                log.write(log_str)
                log.write(str(GL_total))
                break

        log.close()
        train_writer.close()
        test_writer.close()


if __name__ == "__main__":
    train_model()
