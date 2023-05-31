#!/usr/bin/env python3
"""Task 8: EM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM
    Args:
        X: np.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
            iterations for the algorithm
        tol: non-negative float containing tolerance of the log likelihood,
            used to determine early stopping i.e. if the difference is less
            than or equal to tol you should stop the algorithm
        verbose: boolean that determines if you should print information about
            the algorithm"""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None

    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None, None

    n, d = X.shape
    pi, m, S = initialize(X, k)

    ll = [0, 0]
    for i in range(iterations):
        g, ll[1] = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, ll[1].round(5)))
        pi, m, S = maximization(X, g)

        if abs(ll[0] - ll[1]) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}"
                      .format(i, ll[1].round(5)))
            break
        ll[0] = ll[1]

    return pi, m, S, g, ll
