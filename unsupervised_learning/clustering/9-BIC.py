#!/usr/bin/env python3
"""Task 9: BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
        Information Criterion
    Args:
        X: np.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters to
            check for (inclusive)
        kmax: positive integer containing the maximum number of clusters to
            check for (inclusive)
        iterations: positive integer containing the maximum number of
            iterations for the EM algorithm
        tol: non-negative float containing the tolerance for the EM algorithm
        verbose: boolean that determines if the EM algorithm should print
            information to the standard output
    Returns: best_k, best_result, ll, b"""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None

    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None

    if type(kmax) is not int or kmax <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if kmin >= kmax:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape
    b = []
    loglikelihoods = []
    results = []
    ks = []
    for k in range(kmin, kmax + 1):
        pi, m, S, _, ll = expectation_maximization(X, k, iterations, tol,
                                                   verbose)
        results.append((pi, m, S))
        ks.append(k)
        loglikelihoods.append(ll)
        # p = (k * d + k * d + 1)
        p = k * d * (d + 1) / 2 + d * k + k - 1
        b.append(p * np.log(n) - 2 * ll)

    b = np.array(b)
    best_k = np.argmin(b)

    return ks[best_k], results[best_k], loglikelihoods, b
