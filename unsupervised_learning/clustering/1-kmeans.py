#!/usr/bin/env python3
"""Task 1: K-means"""
import numpy as np


def initialize(X, k):
    """ Initializes cluster centroids for K-means
    Args:
        X: numpy.ndarray, shape(n, d), data points to be clustered
            n: number of data points
            d: dimension of data points
        k: int, number of clusters
    Returns:
        centroids: numpy.ndarray of shape (k, d) containing the initialized
            centroids for each cluster,  or None on failure"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    centroids = np.random.uniform(low=mins, high=maxs, size=(k, X.shape[1]))
    return centroids


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset:
    Args:
        X: numpy.ndarray, shape(n, d), data points to be clustered
            n: number of data points
            d: dimension of data points
        k: int, number of clusters
        iterations: int, max number of iterations
    Returns:
        C: numpy.ndarray, shape(k, d), centroid means for each cluster
        clss: numpy.ndarray, shape(n,), index of the cluster in C that
            each data point belongs to"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    Cs = initialize(X, k)
    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - Cs, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.empty_like(Cs)
        for i in range(k):
            assigned_points = X[clss == i]
            if len(assigned_points) > 0:
                new_C[i] = np.mean(assigned_points, axis=0)
            else:
                new_C[i] = initialize(X, 1)

        if np.allclose(Cs, new_C):
            break

        Cs = new_C

    return Cs, clss
