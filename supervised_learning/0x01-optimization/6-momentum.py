#!/usr/bin/env python3
"""Task 6"""
import numpy as np
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training op for a NN in tensorflow using GD with
    momentum optimization algorithm:
    loss: the loss of the network
    alpha: the learning rate
    beta1: momentum weight

    Returns: the momentum optimiztion operation"""

    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
