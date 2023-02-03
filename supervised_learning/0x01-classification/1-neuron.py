#!/usr/bin/env python3
"""Task 1"""
import numpy as np


class Neuron():
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """initializes neuron, nx is number of input features"""

        self.__A = 0
        self.__b = 0

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=[1, nx])

    @property
    def A(self):
        """Getter for A"""
        return self.__A

    @property
    def b(self):
        """Getter for b"""
        return self.__b

    @property
    def W(self):
        """Getter for W"""
        return self.__W
