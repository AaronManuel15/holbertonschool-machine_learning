#!/usr/bin/env python3
"""Task 2"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix
    Args:
        labels: a numpy.ndarray with shape (m,) containing numeric class labels
        classes: the maximum number of classes found in labels
    Conditions:
        The last dimension of the one-hot matrix must be the number of classes
    Returns:
        one-hot matrix"""

    return K.utils.to_categorical(labels, classes)
