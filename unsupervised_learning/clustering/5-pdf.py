#!/usr/bin/env python3
"""Task 5: PDF"""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution
    Args:
        X: np.ndarray of shape (n, d) containing the data points whose PDF
            should be evaluated
        m: np.ndarray of shape (d,) containing the mean of the distribution
        S: np.ndarray of shape (d, d) containing the covariance of the
            distribution
    Returns:
        P: np.ndarray of shape (n,) containing the PDF values for each data
            point"""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d:
        return None

    if S.shape[0] != d or S.shape[1] != d:
        return None

    inv_S = np.linalg.inv(S)
    constant = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(S))

    # Broadcasting to subtract mean from each row of X
    diff = X - m.reshape(1, d)
    exponent = -0.5 * np.sum(np.dot(diff, inv_S) * diff, axis=1)
    P = constant * np.exp(exponent)

    return P
