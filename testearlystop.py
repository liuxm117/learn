# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta


# 记录训练花费的时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    # timedelta是用于对间隔进行规范化输出，间隔10秒的输出为：00:00:10
    return timedelta(seconds=int(round(time_dif)))


n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


# 打乱数据，并生成batch
def shuffle_batch(X, y, batch_size):
    # permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    # 把rnd_idx这个一位数组进行切分
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)

# 定义损失函数和计算损失
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# 定义优化器
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# 评估模型，使用准确性作为我们的绩效指标
with tf.name_scope("eval"):
    # logists最大值的索引在0-9之间，恰好就是被预测所属于的类，因此和y进行对比，相等就是True，否则为False
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 定义好训练轮次和batch-size
n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    start_time = time.time()

    # 记录总迭代步数，一个batch算一步
    # 记录最好的验证精度
    # 记录上一次验证结果提升时是第几步。
    # 如果迭代2000步后结果还没有提升就中止训练。
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 2000

    flag = False
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            # 每次迭代10步就验证一次
            if total_batch % 10 == 0:
                acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})

                # 如果验证精度提升了，就替换为最好的结果，并保存模型
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    save_path = saver.save(sess, "./my_model_stop.ckpt")
                    improved_str = 'improved!'
                else:
                    improved_str = ''
                # 记录训练时间，并格式化输出验证结果。
                time_dif = get_time_dif(start_time)
                msg = 'Epoch:{0:>4}, Iter: {1:>6}, Acc_Train: {2:>7.2%}, Acc_Val: {3:>7.2%}, Time: {4} {5}'
                print(msg.format(epoch, total_batch, acc_batch, acc_val, time_dif, improved_str))

            # 记录总迭代步数
            total_batch += 1

            # 如果2000步以后还没提升，就中止训练。
            if total_batch - last_improved > require_improvement:
                print("No optimization for ", require_improvement, " steps, auto-stop in the ", total_batch, " step!")
                # 跳出这个轮次的循环
                flag = True
                break
        # 跳出所有训练轮次的循环
        if flag:
            break

with tf.Session() as sess:
    saver.restore(sess, "./my_model_stop.ckpt")  # or better, use save_path
    X_test_20 = X_test[:20]
    # 得到softmax之前的输出
    Z = logits.eval(feed_dict={X: X_test_20})
    # 得到每一行最大值的索引
    y_pred = np.argmax(Z, axis=1)
    print("Predicted classes:", y_pred)
    print("Actual calsses:   ", y_test[:20])
    # 评估在测试集上的正确率
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("\nTest_accuracy:", acc_test)