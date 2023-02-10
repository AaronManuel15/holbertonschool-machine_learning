#!/usr/bin/env python3
"""Task 0: creates placeholders for NN"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """ nx: the number of feature columns in our data
        classes: the number of classes in our classifier"""

    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
