#!/usr/bin/env python3
"""Task 1. RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN
    Args:
        rnn_cell: instance of RNNCell that will be used for the forward
            propagation
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state, given as a numpy.ndarray of shape (m, h)
            h: dimensionality of the hidden state
    Returns:
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs"""
    t, m, _ = X.shape
    _, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])
    return H, Y
