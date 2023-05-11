#!/usr/bin/env python3
"""Task 5: Definiteness"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 2 and matrix.shape[0] != matrix.shape[1] or \
            len(matrix.shape) != 2:
        return None
    if np.allclose(matrix, matrix.T) is False:
        return None
    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"
    if np.all(np.linalg.eigvals(matrix) >= 0):
        return "Positive semi-definite"
    if np.all(np.linalg.eigvals(matrix) < 0):
        return "Negative definite"
    if np.all(np.linalg.eigvals(matrix) <= 0):
        return "Negative semi-definite"
    return "Indefinite"
