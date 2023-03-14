#!/usr/bin/env python3
"""Task 4"""
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5 architecture using TF
    Args:
        x: tf.placeholder of shape (m, 28, 28, 1) containing the input
            images for the network
            m: number of images
        y; tf.placeholder of shape (m, 10) containing the one-hot labels
            for the network"""

    # kernel initializer
    init = tf.contrib.layers.variance_scaling_initializer()

    # convolutional layer 1
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                             padding='same', activation=tf.nn.relu,
                             kernel_initializer=init)(x)

    # pooling layer 1
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)

    # convolutional layer 2
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                             padding='valid', activation=tf.nn.relu,
                             kernel_initializer=init)(pool1)

    # pooling layer 2
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)

    # flatten
    flatten = tf.layers.Flatten()(pool2)

    # fully connected layer 1
    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                          kernel_initializer=init)(flatten)

    # fully connected layer 2
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                          kernel_initializer=init)(fc1)

    # fully connected layer 3 aka prediction layer
    y_pred = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)

    # loss of the network
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # train operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # accuracy of the nextwork
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1),
                                          tf.argmax(y, 1)), tf.float32))

    # softmax activation
    softmax = tf.nn.softmax(y_pred)

    return softmax, train_op, loss, acc
