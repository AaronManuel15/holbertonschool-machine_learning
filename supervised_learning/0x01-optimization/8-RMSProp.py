#!/usr/bin/env python3
"""Task 8"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training op for NN in tf using RMSProp Opt Algo:
    loss: loss of the NN
    alpha: the learning rate
    beta2: the RMSProp weight
    epsilon: small number to avoid division by zero

    Returns: the RMSProp optimization operation"""

    return tf.train.RMSPropOptimizer(learning_rate=alpha, epsilon=epsilon,
                                     momentum=beta2).minimize(loss)
