#!/usr/bin/env python3
"""Task 1: Regular Chains"""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain
    Args:
        P: is a square 2D numpy.ndarray of shape (n, n) representing the
            transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns:
        a numpy.ndarray of shape (1, n) representing the steady state
        probabilities, or None on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.any(P <= 0):
        return None

    n = P.shape[0]
    s = np.ones((1, n)) / n
    t = 1000
    for i in range(t):
        s = s @ P
    return s
