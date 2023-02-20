#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization constants of a matrix
    X: a numpy.ndarray of shape (m, nx)
    where m is number of data points and
    nx is the number of features
    Returns the mean and standard dev of each feature respectively"""

    mean, stddev = [], []
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)

    return mean, stddev
