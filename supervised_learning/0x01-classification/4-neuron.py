#!/usr/bin/env python3
"""Task 4"""
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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""

        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1+np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""

        return (-np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A)))/(A.shape[1])

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions"""

        return (np.rint(self.forward_prop(X)).astype(int),
                self.cost(Y, self.__A))
