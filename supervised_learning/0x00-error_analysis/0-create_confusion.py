#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix:

    labels: one-hot numpy.ndarray of shape (m, classes) containing the
        correct labels for each data point
        m: is number of data points
        classes: number of classes
    logits: one-hot numpy.ndarray of shape (m, classes) containing the
        predicted labels

    Returns: a confusion numpy.ndarray of shape (classes, classes)
        with row indices representing the correct labels and column
        indices representing the predicted labels"""

    confusion = np.zeros((len(labels[1]), len(logits[1])))
    labelsT, logitsT = np.where(labels == 1)[1], np.where(logits == 1)[1]
    for x in range(len(labelsT)):
        confusion[labelsT[x]][logitsT[x]] += 1
    return confusion
