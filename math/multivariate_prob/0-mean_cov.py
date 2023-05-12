#!/usr/bin/env python3
"""Task 0: Mean and Covariance"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set:
    Args:
        X (np.ndarray): shape of (n, d) where
        n is the number of data points
        d is the number of dimensions in each data point"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = [np.mean(X, axis=0)]
    cov = np.matmul((X - mean).T, (X - mean)) / (X.shape[0] - 1)
    return mean, cov
