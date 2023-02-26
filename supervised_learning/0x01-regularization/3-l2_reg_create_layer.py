#!/usr/bin/env python3
"""Task 3"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization:

    prev: is a tensor containing the output of the previous layer
    n: is the number of nodes the new layer should contain
    activation: is the activation function that should be used
        on the layer
    lambtha: is the L2 regularization parameter

    Returns: the output of the new layer"""

    K_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    K_reg = tf.contrib.layers.l2_regularizer(lambtha)
    l2_layer = tf.layers.Dense(units=n,
                               activation=activation,
                               kernel_initializer=K_init,
                               kernel_regularizer=K_reg)(prev)
    return l2_layer
