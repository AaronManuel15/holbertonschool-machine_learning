#!/usr/bin/env python3
"""Task 14"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch norm layer for a NN in TF:
    prev: is the activated output of the previous layer
    n: is the number of nodes in the layer to be created
    activation: is the activation function that should be
        used on the output of the layer

    Returns: a tensor of the activated output for the layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, kernel_initializer=init)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    u, var = tf.nn.moments(layer, 0)
    return activation(tf.nn.batch_normalization(x=layer,
                                                mean=u,
                                                variance=var,
                                                offset=beta,
                                                scale=gamma,
                                                variance_epsilon=1e-8))
