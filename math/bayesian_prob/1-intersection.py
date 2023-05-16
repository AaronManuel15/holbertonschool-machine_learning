#!/usr/bin/env python3
"""Task 1: Intersection"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.
    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing the various hypothetical probabilities
            of developing severe side effects
        Pr: 1D numpy.ndarray containing the prior beliefs of P"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal" +
                         " to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if min(P) < 0 or max(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    if min(Pr) < 0 or max(Pr) > 1:
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    fact = np.math.factorial
    comb = fact(n) / (fact(x) * fact(n - x))
    return Pr * comb * (P ** x) * ((1 - P) ** (n - x))
