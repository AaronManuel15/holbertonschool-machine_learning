#!/usr/env python3
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
