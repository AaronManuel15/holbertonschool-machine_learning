#!/usr/bin/env python3
"""Task 11: GMM"""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset
    Args:
        X: np.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
    Returns: pi, m, S, clss, bic"""

    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
