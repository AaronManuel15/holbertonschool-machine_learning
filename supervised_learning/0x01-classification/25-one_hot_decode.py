#!/usr/bin/env python3
"""Task 25"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels:

    one_hot is assumed to be a numpy.ndarray with shape (classes, m)
    Classes is the maximum number of classes
    m is the number of examples
    Returns: a numpy.ndarray with shape (m,) or None"""

    if type(one_hot) is not np.ndarray:
        return None
    m = len(one_hot)
    try:
        LabelVector = np.argmax(one_hot.T, axis=1)
        return LabelVector
    except Exception:
        return None
