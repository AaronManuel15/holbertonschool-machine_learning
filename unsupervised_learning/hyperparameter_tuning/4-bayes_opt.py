#!/usr/bin/env python3
"""Task 3. Initialize Bayesian Optimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """Constructor to set instance attributes
        Args:
            f: black-box function to be optimized
            X_init: np.ndarray of shape (t, 1) representing the inputs
                already sampled with the black-box function
            Y_init: np.ndarray of shape (t, 1) representing the outputs
                t: number of initial samples
            bounds: tuple of (min, max) representing the bounds of the space
                to look for the optimal point
            ac_samples: number of samples that should be analyzed during
                acquisition
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output of the
                black-box function
            xsi: exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculated the next best sample location with Expected
        Improvement acquisition function
        Returns:
            X_next: np.ndarray of shape (1,) representing the next best
                sample point
            EI: np.ndarray of shape (ac_samples,) containing the expected
                improvement of each potential sample"""

        from scipy.stats import norm
        mu, _ = self.gp.predict(self.gp.X)
        mu_sample, sigma_sample = self.gp.predict(self.X_s)

        # Defining
        if self.minimize is True:
            mu_sample_opt = np.min(mu)
        else:
            mu_sample_opt = np.max(mu)

        imp = mu_sample_opt - mu_sample - self.xsi
        Z = imp / sigma_sample
        ei = imp * norm.cdf(Z) + sigma_sample * norm.pdf(Z)

        X_next = self.X_s[np.argmax(ei)]

        return X_next, np.array(ei)
