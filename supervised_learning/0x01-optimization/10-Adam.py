#!/usr/bin/env python3
"""Task 10"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training op for NN in TF using Adam Opt Algo:
    loss: is the loss of the network
    alpha: is the learning rate
    beta1: is the weight used for the first moment
    beta2: is the weight used for the second moment
    epsilon: is a small number to avoid division by zero

    Returns: the Adam optimization operation"""

    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1,
                                  beta2=beta2,
                                  epsilon=epsilon).minimize(loss)
