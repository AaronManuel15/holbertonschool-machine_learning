#!/usr/bin/env python3
"""Task 9"""
import numpy as np


class NeuralNetwork():
    """Defines a neural network network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):
        """initializes network,
        nx is number of input features
        nodes is the number of nodes in the hidden layer"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=[nodes, nx])
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=[1, nodes])
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2"""
        return self.__A2
