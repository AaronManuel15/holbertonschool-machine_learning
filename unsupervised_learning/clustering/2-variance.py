#!/usr/bin/env python3
"""Task 2: Variance"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set
    Args:
        X: numpy.ndarray, shape(n, d), data points being clustered
            n: number of data points
            d: dimension of data points
        C: numpy.ndarray, shape(k, d), centroid means for each cluster
    Returns:
        var: the total variance"""

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return np.sum(np.square(distances[np.arange(len(X)), clss]))
