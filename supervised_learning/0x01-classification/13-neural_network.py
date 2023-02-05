#!/usr/bin/env python3
"""Task 13"""
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

    def forward_prop(self, X):
        """Calculates the forward propagation of the Neural Network"""

        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1+np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1+np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""

        return (-np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A)))/(A.shape[1])

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""

        self.__A1, self.__A2 = self.forward_prop(X)
        return (np.rint(self.__A2).astype(int),
                self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the NN"""

        dz2 = A2 - Y
        dW2 = np.dot(dz2, A1.T)/dz2.shape[1]
        db2 = np.sum(dz2, axis=1, keepdims=True)/Y.shape[1]
        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))
        dW1 = np.dot(dz1, X.T)/Y.shape[1]
        db1 = np.sum(dz1, axis=1, keepdims=True)/Y.shape[1]

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
