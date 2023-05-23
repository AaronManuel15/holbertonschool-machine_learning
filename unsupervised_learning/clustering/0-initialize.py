#!/usr/bin/env python3
"""Task 0: Initialize K-means"""
import numpy as np


def initialize(X, k):
    """ Initializes cluster centroids for K-means
    Args:
        X: numpy.ndarray, shape(n, d), data points to be clustered
            n: number of data points
            d: dimension of data points
        k: int, number of clusters"""
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    centroids = np.random.uniform(low=mins, high=maxs, size=(k, X.shape[1]))
    return centroids
