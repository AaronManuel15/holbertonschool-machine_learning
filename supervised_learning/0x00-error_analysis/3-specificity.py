#!/usr/bin/env python3
"""Task 3"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column
        indices represent the predicted labels
    classes: is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing the
        sensitivity of each class"""

    tp = np.diag(confusion)
    tpNfp = np.sum(confusion, axis=0)
    tpNfn = np.sum(confusion, axis=1)
    tn = np.sum(confusion) - tpNfn - tpNfp + tp
    return tn / (tn + tpNfp - tp)
