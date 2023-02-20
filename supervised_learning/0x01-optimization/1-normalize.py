#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def normalize(X, m, s):
    """Normalizes a matrix
    X: a numpy.ndarray of shape (m, nx) to normalize
    where m is number of data points and
    nx is the number of features
    m: array of shape (nx,) of means of all X features
    s: array of shape (nx,) sttdev of all X features
    Returns the normalized X matrix"""

    X = (X - m)/s
    return X
