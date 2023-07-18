#!/usr/bin/env python3
"""Task 5. Bidirectional Cell"""
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
        self.Wy = np.random.normal(size=(h, o))
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
