# coding:utf-8
#!/usr/bin/python

import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
X_holder = tf.placeholder(tf.float32)
y_holder = tf.placeholder(tf.float32)


def t0():
    c = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
    with tf.Session() as sess:
        print(sess.run(c))
    pass


def mnist_show_image():
    image = mnist.train.images[998].reshape(-1, 28)
    plt.subplot(131)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.show()
    print('\n')


def drawDigit(position, image, title):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    plt.title(title)


def batchDraw():
    batch_size = 196
    images, labels = mnist.train.next_batch(batch_size)
    image_number = images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number, column_number))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index+1)
                image = images[index]
                title = 'actual:%d' % (np.argmax(labels[index]))
                drawDigit(position, image, title)
    plt.show()


def drawDigit2(position, image, title, isTrue):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    if not isTrue:
        plt.title(title, color='red')
    else:
        plt.title(title)


def batchDraw2(batch_size, session, predict_y):
    images, labels = mnist.test.next_batch(batch_size)
    predict_labels = session.run(
        predict_y, feed_dict={X_holder: images, y_holder: labels})
    image_number = images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number+8, column_number+8))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index+1)
                image = images[index]
                actual = np.argmax(labels[index])
                predict = np.argmax(predict_labels[index])
                isTrue = actual == predict
                title = 'actual:%d\npredict:%d' % (actual, predict)
                drawDigit2(position, image, title, isTrue)
    plt.show()


def mnist_t1():
    print(dir(mnist)[-10:])
    print(mnist.train.num_examples)
    print(mnist.validation.num_examples)
    print(mnist.test.num_examples)

    Weights = tf.Variable(tf.zeros([784, 10]))
    biases = tf.Variable(tf.zeros([1, 10]))

    predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)
    loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(500):
        images, labels = mnist.train.next_batch(batch_size)
        session.run(train, feed_dict={X_holder: images, y_holder: labels})
        #print(session.run(predict_y, feed_dict={X_holder: images}))
        #print(session.run(y_holder, feed_dict={y_holder: labels}))
        # print('\n\n\n')
        if i % 25 == 0:
            correct_prediction = tf.equal(
                tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_value = session.run(
                accuracy, feed_dict={X_holder: mnist.test.images, y_holder: mnist.test.labels})
            print('step:%d accuracy:%.4f' % (i, accuracy_value))

    #batchDraw2(batch_size, session, predict_y)


def addConnect(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal(
        [in_size, out_size], stddev=0.01))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)


def mnist_t2():
    connect_1 = addConnect(X_holder, 784, 300, tf.nn.relu)
    predict_y = addConnect(connect_1, 300, 10, tf.nn.softmax)
    loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
    optimizer = tf.train.AdagradOptimizer(0.3)
    train = optimizer.minimize(loss)

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(1000):
        images, labels = mnist.train.next_batch(batch_size)
        session.run(train, feed_dict={X_holder: images, y_holder: labels})
        if i % 50 == 0:
            correct_prediction = tf.equal(
                tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_value = session.run(
                accuracy, feed_dict={X_holder: mnist.test.images, y_holder: mnist.test.labels})
            print('step:%d accuracy:%.4f' % (i, accuracy_value))


def mnist_t3():
    X_images = tf.reshape(X_holder, [-1, 28, 28, 1])
    # convolutional layer 1
    conv1_Weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    conv1_biases = tf.Variable(tf.constant(0.1, shape=[32]))
    conv1_conv2d = tf.nn.conv2d(X_images, conv1_Weights, strides=[
        1, 1, 1, 1], padding='SAME') + conv1_biases
    conv1_activated = tf.nn.relu(conv1_conv2d)
    conv1_pooled = tf.nn.max_pool(conv1_activated, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # convolutional layer 2
    conv2_Weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2_conv2d = tf.nn.conv2d(conv1_pooled, conv2_Weights, strides=[
        1, 1, 1, 1], padding='SAME') + conv2_biases
    conv2_activated = tf.nn.relu(conv2_conv2d)
    conv2_pooled = tf.nn.max_pool(conv2_activated, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # full connected layer 1
    connect1_flat = tf.reshape(conv2_pooled, [-1, 7 * 7 * 64])
    connect1_Weights = tf.Variable(
        tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    connect1_biases = tf.Variable(tf.constant(0.1, shape=[1024]))
    connect1_Wx_plus_b = tf.add(
        tf.matmul(connect1_flat, connect1_Weights), connect1_biases)
    connect1_activated = tf.nn.relu(connect1_Wx_plus_b)
    # full connected layer 2
    connect2_Weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    connect2_biases = tf.Variable(tf.constant(0.1, shape=[10]))
    connect2_Wx_plus_b = tf.add(
        tf.matmul(connect1_activated, connect2_Weights), connect2_biases)
    predict_y = tf.nn.softmax(connect2_Wx_plus_b)
    # loss and train
    loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
    #optimizer = tf.train.AdamOptimizer(0.0001)
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    for i in range(1001):
        train_images, train_labels = mnist.train.next_batch(200)
        session.run(train, feed_dict={
            X_holder: train_images, y_holder: train_labels})
        if i % 100 == 0:
            correct_prediction = tf.equal(
                tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy/t3', accuracy)
            test_images, test_labels = mnist.test.next_batch(2000)
            train_accuracy = session.run(
                accuracy, feed_dict={X_holder: train_images, y_holder: train_labels})
            test_accuracy = session.run(
                accuracy, feed_dict={X_holder: test_images, y_holder: test_labels})
            print('step:%d train accuracy:%.4f test accuracy:%.4f' %
                  (i, train_accuracy, test_accuracy))


if __name__ == '__main__':
    # t0()
    # mnist_show_image()
    # batchDraw()
    mnist_t1()
    # mnist_t2()
    # mnist_t3()
