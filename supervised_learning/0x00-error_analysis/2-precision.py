#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column
        indices represent the predicted labels
    classes: is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing the
        sensitivity of each class"""

    prec = np.zeros(len(confusion[0]))
    for x in range(len(confusion)):
        prec[x] = confusion[x][x] / confusion[:, x].sum()

    return prec
