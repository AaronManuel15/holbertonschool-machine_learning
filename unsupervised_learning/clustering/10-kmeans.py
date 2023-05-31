#!/usr/bin/env python3
"""Task 10: Hello, sklearn!"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset
    Args:
        X: np.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
    Returns: C, clss"""

    C, clss, _ = sklearn.cluster.k_means(X, k)
    return C, clss
