#!/usr/bin/env python3
"""Task 2: Initialize Gaussian Process"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """constructor"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data.T, axis=0)[np.newaxis, :].T
        self.cov = np.matmul((data.T - self.mean.T).T,
                             (data.T - self.mean.T)) / (data.shape[1] - 1)

    def pdf(self, x):
        """Calcluates the PDF at a data point:
        Args:
            x (np.ndarray): shape of (d, 1) containing the data point
            whose PDF should be calculated
            d is the number of dimensions of the Multinomial instance"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError("x must have the shape ({d}, 1)")
        d = self.cov.shape[0]
        x_m = x - self.mean
        Px = 1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))
        Px *= np.exp(-0.5 * np.matmul(np.matmul((x_m).T,
                                                np.linalg.inv(self.cov)),
                                      (x_m)))
        return Px[0][0]
