#!/usr/bin/env python3
"""Task 1: PCAv2"""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset
    Args:
        X: numpy.ndarray of shape (n, d)
            n: number of data points
            d: number of dimensions
        ndim: new dimensionality of the transformed X
    Returns:
        T: numpy.ndarray of shape (n, ndim) contining the
        transformed version"""
    mean = np.mean(X, axis=0)
    X_m = (X - mean)
    U, S, Vt = np.linalg.svd(X_m)
    W = Vt[:ndim].T
    T = np.dot(X_m, W)
    return T
