#!/usr/bin/env python3
"""Task 2: Absorbing Chains"""
import numpy as np


def absorbing(P):
    """Determines if a makov chain is absorbing
    Args:
        P: is a square 2D numpy.ndarray of shape (n, n) representing the
            transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns:
        True if it is absorbing, or False on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if np.sum(P, axis=1).all() != 1:
        return False
    if not np.any(np.diag(P) == 1):
        return False
    if (np.diag(P) == 1).all():
        return True

    return False
