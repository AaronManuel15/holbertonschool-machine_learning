#!/usr/bin/env python3
"""Task 5: creates the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """loss: the loss of the network's prediction
    alpha: the learning rate"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)

    return optimizer.minimize(loss)
