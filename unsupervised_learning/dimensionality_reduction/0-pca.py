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
    # Step 1: Calculate the SVD of the input data
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Step 2: Calculate the cumulative explained variance ratio
    explained_variance_ratio = np.cumsum(S**2) / np.sum(S**2)

    # Step 3: Determine the number of principal components
    n_components = np.argmax(explained_variance_ratio >= var) + 1

    # Step 4: Select the first n_components right singular vectors
    W = Vt[:n_components + 1].T
    return W
