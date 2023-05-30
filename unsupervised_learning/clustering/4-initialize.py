#!/usr/bin/env python3
"""Task 4: Initialize"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model
    Args:
        X: np.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
    Returns:
        pi: np.ndarray of shape (k,) containing the priors for each cluster,
            initialized evenly
        m: np.ndarray of shape (k, d) containing the centroid means for each
            cluster, initialized with K-means
        S: np.ndarray of shape (k, d, d) containing the covariance matrices
            for each cluster, initialized as identity matrices"""

    try:
        m, _ = kmeans(X, k)
        pi = np.ones((k,)) / k

        _, d = X.shape

        S = np.zeros((k, d, d))
        S[:] = np.eye(d)

        return pi, m, S

    except Exception:
        return None, None, None
