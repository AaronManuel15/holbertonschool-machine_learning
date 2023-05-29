#!/usr/bin/env python3
"""Task 3: Optimum"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters to
            check for (inclusive)
        kmax: positive integer containing the maximum number of clusters to
            check for (inclusive)
        iterations: positive integer containing the maximum number of
            iterations for K-means
    Returns:
        results: list containing the outputs of K-means for each cluster size
        d_vars: list containing the difference in variance from the smallest
            cluster size for each cluster size"""

    if type(kmin) is not int or kmin < 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return None, None

    results, d_vars = [], []
    start, _ = kmeans(X, kmin, iterations)
    var = variance(X, start)
    while(kmin <= kmax):
        klusters, klss = kmeans(X, kmin, iterations)
        results.append((klusters, klss))
        d_vars.append(abs(var - variance(X, klusters)))
        var = variance(X, klusters)
        kmin += 1

    return results, d_vars
