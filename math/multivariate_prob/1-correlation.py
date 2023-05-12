#!/usr/bin/env python3
"""Task 1: Correlation"""
import numpy as np


def correlation(C):
    """Calculates a correlation matrix:
    Args:
        C (np.ndarray): covariance matrix with shape of (d, d) where
        d is the number of dimensions"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    var = np.sqrt(np.diag(C))
    return np.divide(C, np.outer(var, var))
