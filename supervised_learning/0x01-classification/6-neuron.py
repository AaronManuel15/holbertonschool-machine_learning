#!/usr/bin/env python3
"""Task 6"""
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""

        dz = A - Y
        dW = np.dot(X, dz.T)/dz.shape[1]
        self.__W -= alpha * dW.reshape(self.__W.shape)
        self.__b -= alpha * np.sum(dz)/dz.shape[1]

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
