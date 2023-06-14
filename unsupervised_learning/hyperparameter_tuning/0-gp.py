#!/usr/bin/env python3
"""Task 0. Initialize Gaussian Process"""
import numpy as np


class GaussianProcess:
    """Represents a noiselsess 1D Gaussian process"""

    def __init__(self, X_init, Y_init, ll=1, sigma_f=1):
        """Constructor
        Args:
                X_init: np.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
                Y_init: np.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
                t: number of initial samples
                l: length parameter for the kernel
                sigma_f: standard deviation given to the output of the
                    black-box function"""
        self.X = X_init
        self.Y = Y_init
        self.ll = ll
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calclulates the covariance kernel matrix between two matrices"""

        sqdist = np.sum(X1**2, 1).reshape(-1, 1) +\
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
