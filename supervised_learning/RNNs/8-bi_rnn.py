#!/usr/bin/env python3
"""Task 8. Bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward prop for a birectional RNN
    Args:
        bi_cell: instance of BidirectionalCell
        X: data to be used, given as np.ndarray of shape (t, m, i)
            t: maximum number of steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state in the forward direction given as np.
            ndarray of shape (m, h)
            h: dimensionality of the hidden state
        h_t: initial hidden state in the backward direction given as np.
            ndarray of shape (m, h)
            h: dimensionality of the hidden state
    Returns:
        H: np.ndarray containing all of the concatenated hidden states
        Y: np.ndarray containing all of the outputs"""

    t, m, _ = X.shape
    _, h = h_0.shape
    H_forward = np.zeros((t + 1, m, h))
    H_backward = np.zeros((t + 1, m, h))
    H_forward[0] = h_0
    H_backward[0] = h_t
    X_flip = np.flipud(X)
    for step in range(t):
        H_forward[step + 1] = bi_cell.forward(H_forward[step], X[step])
        H_backward[step + 1] = bi_cell.backward(H_backward[step],
                                                X_flip[step])

    H = np.concatenate((H_forward[1:], H_backward[7:0:-1]), axis=2)

    Y = bi_cell.output(H)
    return H, Y
