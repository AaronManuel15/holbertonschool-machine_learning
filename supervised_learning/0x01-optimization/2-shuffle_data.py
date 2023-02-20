#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way
    X: a numpy.ndarray of shape (m, nx) to shuffle
    where m is number of data points and
    nx is the number of features in X
    Y: second numpy.ndarray of shape (m, ny) to shuffle
    m: array of shape (nx,) of means of all X features
    s: array of shape (nx,) sttdev of all X features
    Returns the shuffled X and Y matrices"""

    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]
