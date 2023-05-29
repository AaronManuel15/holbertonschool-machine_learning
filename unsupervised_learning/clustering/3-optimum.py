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

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    results, d_vars = [], []
    k = kmin
    while(k <= kmax):
        klusters, klss = kmeans(X, k, iterations)
        if k == kmin:
            var = variance(X, klusters)
            results.append((klusters, klss))
            d_vars.append(0.0)
            continue

        results.append((klusters, klss))
        d_vars.append(abs(var - variance(X, klusters)))
        k += 1

    return results, d_vars
