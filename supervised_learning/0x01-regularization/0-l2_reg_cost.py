#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a NN with L2 regularization:

    cost: is the cost of the network without L2 regularization
    lambtha: is the regularization parameter
    weights: is a dictionary of the weights and biases (numpy.ndarrays)
        of the neural network
    L: is the number of layers in the neural network
    m: is the number of data points used

    Returns: the cost of the network accounting for L2 regularization"""

    w2squared = 0
    for key, array in weights.items():
        if 'W' in key:
            w2squared += np.sum((array**2))

    return cost + ((lambtha/(2*m)) * w2squared)
