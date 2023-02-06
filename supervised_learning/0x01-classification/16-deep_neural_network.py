#!/usr/bin/env python3
"""Task 16"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural network performing binary classification"""

    def __init__(self, nx, layers):
        """--- Constructor ---
        nx: number of input features
        layers: list representing number of nodes in each layer"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) <= 0:
            TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prevLayer = nx

        for l in range(self.L):
            w = np.random.randn(layers[l], prevLayer)*np.sqrt(2/prevLayer)
            self.weights[print("W{}").format(l + 1)] = w
            self.weights[print("b{}").format(l + 1)] = np.zeros((layers[l], 1))
            prevLayer = layers[l]
