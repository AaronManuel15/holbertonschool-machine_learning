#!/usr/bin/env python3
"""Task 5"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a NN with Dropout Reg using GD:

    Y: one-hot numpy.ndarray of shape (classes, m) that contains the
        correct labels for the data
    classes: is the number of classes
    m: is the number of data points
    weights: is a dictionary of the weights and biases of the
        neural network
    cache: is a dictionary of the outputs and dropout masks
        of each layer of the neural network
    alpha: is the learning rate

    keep_prob: is the probability that a node will be kept
    L: is the number of layers of the network

    All layers use the tanh activation function except the last,
        which uses the softmax activation function
    The weights of the network will be updated in place"""

    m = Y.shape[1]
    for layer in range(L, 0, -1):
        ACurLayer = cache["A{}".format(layer)]
        APrevLayer = cache["A{}".format(layer - 1)]
        if layer == L:
            dz = (ACurLayer - Y)
        else:
            dz = dAPrevLayer * (1 - np.square(ACurLayer))
        W = weights["W{}".format(layer)]
        dW = np.matmul(dz, APrevLayer.T)/m
        db = np.sum(dz, axis=1, keepdims=True)/m
        if layer > 1:
            mask = cache["D{}".format(layer - 1)]
            dAPrevLayer = np.matmul(W.T, dz)*mask
            dAPrevLayer /= keep_prob
        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db
