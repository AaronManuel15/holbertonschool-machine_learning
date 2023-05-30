#!/usr/bin/env python3
"""Task 7: Maximization"""
import numpy as np


def maximization(X, g):
    """Calculats the maximization step in the EM algorithm for a GMM
    Args:
        X: np.ndarray of shape (n, d) containing the data set
        g: np.ndarray of shape (k, n) containing the posterior probabilities
            for each data point in each cluster
    Returns:
        pi: np.ndarray of shape (k,) containing the updated priors for each
            cluster
        m: np.ndarray of shape (k, d) containing the updated centroid means
            for each cluster
        S: np.ndarray of shape (k, d, d) containing the updated covariance
            matrices for each cluster"""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if False in np.isclose(g.sum(axis=0), np.ones((g.shape[1]))):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    pi = np.sum(g, axis=1) / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m[i] = np.sum(g[i].reshape(-1, 1) * X, axis=0) / np.sum(g[i])
        diff = X - m[i]
        S[i] = np.dot(g[i].reshape(1, -1) * diff.T, diff) / np.sum(g[i])

    return pi, m, S
