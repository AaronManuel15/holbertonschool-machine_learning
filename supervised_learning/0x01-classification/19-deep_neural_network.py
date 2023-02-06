#!/usr/bin/env python3
"""Task 19"""
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
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prevLayer = nx

        for l in range(len(layers)):
            if type(layers[l]) is not int:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layers[l], prevLayer)*np.sqrt(2/prevLayer)
            self.weights["W{}".format(l + 1)] = w
            self.weights["b{}".format(l + 1)] = np.zeros((layers[l], 1))
            prevLayer = layers[l]

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation of the NN"""

        self.__cache["A0"] = X
        for l in range(1, self.__L + 1):
            W = self.__weights["W{}".format(l)]
            A = self.__cache["A{}".format(l - 1)]
            b = self.__weights["b{}".format(l)]
            z = np.matmul(W, A) + b
            self.__cache["A{}".format(l)] = 1 / (1+np.exp(-z))
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""

        return (-np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A)))/(A.shape[1])
