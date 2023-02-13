#!/usr/bin/env python3
"""Task 1: creates layers of the NN"""

import tensorflow as tf



def create_layer(prev, n, activation):
    """ prev: tensor output of the previous layer
        n: number of nodes in the layer to be created
        activation: the activation function of the layer"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.dense(prev, n, activation, kernal_regularizer = weights)
