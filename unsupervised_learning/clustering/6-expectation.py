#!/usr/bin/env python3
"""Task 6: Expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM
    Args:
        X: np.ndarray of shape (n, d) containing the data set
        pi: np.ndarray of shape (k,) containing the priors for each cluster
        m: np.ndarray of shape (k, d) containing the centroid means for each
            cluster
        S: np.ndarray of shape (k, d, d) containing the covariance matrices
            for each cluster
    Returns:
        g: np.ndarray of shape (k, n) containing the posterior probabilities
            for each data point in each cluster
        l: total log likelihood"""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None

    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None

    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None

    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None
# create a numpy array of zeros with same shape as pi but more depth
    g = np.zeros((k, n))
# iterate through the number of clusters
    for i in range(k):
        P = pdf(X, m[i], S[i])
        g[i] = pi[i] * P
# calculate the log likelihood before normalizing the probabilities
    ll = np.sum(np.log(g))
# normalize the probabilities
    g /= np.sum(g, axis=0)

    return g, ll
