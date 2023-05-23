#!/usr/bin/env python3
"""TAsk 0. PCA"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset
    Args:
        X: numpy.ndarray of shape (n, d)
            n: number of data points
            d: number of dimensions
            all dimensions have a mean of 0 across all data points
        var: fraction of the variance that the PCA should maintain
    Returns:
        W: numpy.ndarray of shape (d, nd) where nd is the new dimensionality
            after PCA"""
    # compute covariance matrix
    cov = np.cov(X.T)
    # compute eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov)
    # sort eigenvalues and eigenvectors
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx] * -1
    # compute new dimensionality
    nd = 0
    for i in range(len(eig_vals)):
        if sum(eig_vals[:i]) / sum(eig_vals) >= var:
            nd = i
            break
    # compute W
    W = eig_vecs[:, :nd + 1]
    return W
