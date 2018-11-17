# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import model
import sys, time
from tqdm import tqdm # python的进度条库
from tensorflow.examples.tutorials.mnist import input_data
from model import Angular_Softmax_Loss
from model import L2_Sofrmax_Loss
# tf.reset_default_graph()


lr = 0.001 # 学习率不变化
epochs = 1000
batch_size = 256

class VGG19(object):
    def __init__(self, batch_size, num_classes):
        images = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1], name='input')
        labels = tf.placeholder(tf.int64, [batch_size,])
        # images_reshape = tf.reshape(images, [-1, 28, 28, 1])
        embeddings = model.buildCNN(images)
        pred_prob, loss = L2_Sofrmax_Loss(embeddings,labels, num_classes=num_classes, margin=1)
        self.x_ = images
        self.y_ = labels
        self.y = tf.argmax(pred_prob, 1)
        self.embeddings = embeddings
        self.loss = loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_),'float32'))


# prepare data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=False, reshape=False)

# training
# with tf.Session() as sess:



mod = VGG19(batch_size, num_classes)
global_step = tf.Variable(0, trainable=False)
# add_step_op = tf.assign_add(global_step, tf.constant(1))

learningrate = tf.train.exponential_decay(0.001, global_step, 10000, 0.9, staircase=False)
optimizer = tf.train.AdamOptimizer(learningrate)
train_optimizer = optimizer.minimize(mod.loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# process_bar = ShowProcess(epochs, 'OK')

for step in range(epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    _, v = sess.run([train_optimizer, mod.loss], feed_dict={mod.x_: batch_xs, mod.y_:batch_ys})
    if step%100 == 0:
        # print(step, sess.run([train_optimizer, mod.loss], feed_dict={mod.x_: batch_xs, mod.y_:batch_ys}))
        print("Step:", step+100, "\tThe L2-Softmax Loss is:", v)
    max_steps = 100


    # process_bar.show_process()
    time.sleep(0.01)


print("The training step is done")

# evaluating
# with tf.Session() as sess:
# mod = VGG19(batch_size, num_classes)

n_test = mnist.test.images.shape[0]
test_data = np.ndarray([batch_size, 28, 28, 1])
test_labels = np.ndarray([batch_size,])

accuracy_vector = []
# init = tf.global_variables_initializer()
# sess.run(init)
for step1 in range(0, n_test, batch_size):
    each = min([step1 + batch_size, n_test])
    size = each - step1
    test_data[0:size,:] = mnist.test.images[step1: each]
    test_labels[0:size] = mnist.test.labels[step1: each]
    result = sess.run(mod.accuracy, feed_dict={mod.x_: test_data, mod.y_: test_labels})
    accuracy_vector.append(result)
print("The testing step is done")

print('Test Accuracy is:', np.mean(np.array(accuracy_vector)))

