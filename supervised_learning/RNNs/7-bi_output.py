#!/usr/bin/env python3
"""Task 7. Bidirectional Output"""
import numpy as np


class BidirectionalCell:
    """represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """Constructor
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(i + h + o, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data input for
                the cell
                m: batch size for the data
            Returns: h_next, y
                h_next: next hidden state
                y: output of the cell"""
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """calculates the hidden state in the backward direction for one time
        step."""
        h_x = np.concatenate((h_next, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)
        return h_next

    def output(self, H):
        """Calculates all ouputs for the RNN
        Args:
            H: np.ndarray of shape (t, m, 2 * h) that contains the concatenated
                hidden states from both directions, excluding initialized
                t: number of time steps
                m: batch size for the data
                h: dimensionality of the hidden states
        Returns:
            Y: the outputs"""

        Y = H @ self.Wy + self.by
        return np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)
