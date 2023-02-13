#!/usr/bin/env python3
"""Task 3: creates the accuracy of a prediction for the NN"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """y: is a placeholder for the labels of the input data
    y_pred is a tensor containing the network's predictions
    returns: a tensor containing the decimal accuracy of the prediction"""

    y = tf.math.argmax(y, axis=1)
    prediction = tf.math.argmax(y_pred, axis=1)
    equality = tf.math.equal(prediction, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
