#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a NN using GD with L2 Reg:

    Y: is a one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data
    classes: is the number of classes
    m: is the number of data points
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs of each layer of the neural network
    alpha: is the learning rate
    lambtha: is the L2 regularization parameter
    L: is the number of layers of the network

    The neural network uses tanh activations on each layer except the
        last, which uses a softmax activation
    The weights and biases of the network should be updated in place"""

    m = Y.shape[1]
    for layer in range(L, 0, -1):
        ACurLayer = cache["A{}".format(layer)]
        APrevLayer = cache["A{}".format(layer - 1)]
        if layer == L:
            dz = (ACurLayer - Y)
        else:
            dz = dAPrevLayer * (1 - np.square(ACurLayer))
        W = weights["W{}".format(layer)]
        l2 = (lambtha/m) * W
        dW = np.matmul(dz, APrevLayer.T)/Y.shape[1] + l2
        db = np.sum(dz, axis=1, keepdims=True)/Y.shape[1]
        dAPrevLayer = np.matmul(W.T, dz)
        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db
