#!/usr/bin/env python3
"""Task 6"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a tensorflow layer that includes Dropout:

    prev: is a tensor containing the output of the previous layer
    n: is the number of nodes the new layer should contain
    activation: is the activation function that should be used
        on the layer
    keep_prob: probability that a node will be kept

    Returns: the output of the new layer"""

    K_init = tf.contrib.layers.VarianceScaling(mode='FAN_AVG')
    K_reg = tf.layers.Dropout(rate=(1-keep_prob))
    NewLayer = tf.layers.Dense(units=n,
                               activation=activation,
                               kernel_initializer=K_init,
                               kernel_regularizer=K_reg)(prev)
    return NewLayer
