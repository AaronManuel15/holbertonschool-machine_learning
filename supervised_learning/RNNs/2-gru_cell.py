#!/usr/bin/env python3
"""Task 2. GRU"""
import numpy as np


class GRUCell:
    """Gated Recurrent Unit"""
    def __init__(self, i, h, o):
        """Constructor
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step
        Args
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data input for
                the cell
                m: batch size for the data
        Returns:
            h_next: next hidden state
            y: output of the cell"""
        # previous hidden and input data combined
        M = np.concatenate((h_prev, x_t), axis=1)

        # Update Gate z and Reset Gate r
        z_t = self.sigmoid(np.matmul(M, self.Wz) + self.bz)
        r_t = self.sigmoid(np.matmul(M, self.Wr) + self.br)

        # construct new hidden state
        act_in = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_t = np.tanh(act_in @ self.Wh + self.bh)
        h_t = (1 - z_t) * h_prev + z_t * h_t

        # new output calculated and Softmax applied
        y = h_t @ self.Wy + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_t, y

    def sigmoid(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))
