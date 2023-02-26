#!/usr/bin/env python3
"""Task 4"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Droupout:

    X: numpy.ndarray of shape (nx, m) containing input data for the network
        nx: is the number of input features
        m: is the number of data points
    weights: is a dictionary of the weights and biases of the neural network
    L: the number of layers in the network
    keep_prob: is the probability that a node will be kept

    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function

    Returns: a dictionary containing the outputs of each layer
        and the dropout mask used on each layer"""

    cache = {}
    cache["A0"] = X
    for lay in range(1, L + 1):
        W = weights["W{}".format(lay)]
        A = cache["A{}".format(lay - 1)]
        b = weights["b{}".format(lay)]
        z = np.matmul(W, A) + b

        if lay == L:
            softA = np.exp(z)/np.sum(np.exp(z), axis=0)
            cache["A{}".format(lay)] = softA

        else:
            newA = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            d = (np.random.rand(newA.shape[0], newA.shape[1]) < keep_prob)
            cache["D{}".format(lay)] = d.astype(int)
            newA *= d
            newA /= keep_prob
            cache["A{}".format(lay)] = newA

    return cache
