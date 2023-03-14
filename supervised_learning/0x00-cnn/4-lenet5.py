#!/usr/bin/env python3
"""Task 4"""
import numpy as np
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5 architecture using TF
    Args:
        x: tf.placeholder of shape (m, 28, 28, 1) containing the input
            images for the network
            m: number of images
        y; tf.placeholder of shape (m, 10) containing the one-hot labels
            for the network"""

    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                             padding='same', activation=tf.nn.relu,
                             kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                             padding='valid', activation=tf.nn.relu,
                             kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(conv2)
    flatten = tf.layers.Flatten()(pool2)
    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                          kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                          kernel_initializer=init)(fc1)
    y_pred = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    y_pred = tf.argmax(y_pred, 1)
    y_true = tf.argmax(y, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return y_pred, train_op, loss, acc
