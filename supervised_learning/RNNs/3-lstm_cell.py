#!/usr/bin/env python3
"""Task 3. LSTM"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""
    def __init__(self, i, h, o):
        """Constructor for the cell
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell"""

        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((i, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((i, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((i, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((i, h))
        self.Wy = np.random.normal(size=(i + o, o))
        self.by = np.zeros((i, o))

    def forward(self, h_prev, c_prev, x_t):
        """Forward pass of the cell for one step
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                hidden state
            c_prev: numpy.ndarray of shape (m, h) containing the previous
                cell state
            x_t: numpy.ndarray of shape (m, i) containing the data input for
                the cell
                m: batch size for the data
        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell"""

        input = np.concatenate((h_prev, x_t), axis=1)
        # Forget Gate
        f_gate = self.sigmoid(input @ self.Wf + self.bf)
        # Input Gate
        i_gate = self.sigmoid(input @ self.Wu + self.bu)
        c_update = np.tanh(input @ self.Wc + self.bc)
        input_gate = i_gate * c_update
        # Output Gate
        o_gate = self.sigmoid(input @ self.Wo + self.bo)
        # Cell State work
        c_next = c_prev * f_gate + input_gate
        # Next hidden
        h_next = np.tanh(c_next) * o_gate
        # Next Output
        y = h_next @ self.Wy + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y

    def sigmoid(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))
