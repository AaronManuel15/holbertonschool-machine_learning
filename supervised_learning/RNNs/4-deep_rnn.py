#!/usr/bin/env python3
"""Task 4. Deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN
    Args:
        rnn_cells: list of RNNCell instances of length l that will be used
            for the forward propagation
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state, given as a numpy.ndarray of shape (l, m, h)
            h: dimensionality of the hidden state
    Returns:
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs"""

    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = []
    for step in range(t):
        for layer in range(l):
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        Y.append(y)

    return H, np.array(Y)
