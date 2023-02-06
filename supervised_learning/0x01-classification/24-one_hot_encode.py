#!/usr/bin/env python3
"""Task 24"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix:

    Y is assumed to be a numpy.ndarray with shape (m,) containing
    numeric class labels with m being the number of examples
    Classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m) or None"""

    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    m = Y.size
    try:
        OHMatrix = np.zeros([classes, m])
        OHMatrix[Y, np.arange(m)] = 1
        return OHMatrix
    except Exception:
        return None
