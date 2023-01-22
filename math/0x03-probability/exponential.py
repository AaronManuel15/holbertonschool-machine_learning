#!/usr/bin/env python3
"""Task 3 - 5 for 0x03 - Probability"""
pi = 3.1415926536
e = 2.7182818285


class Exponential():
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """initializes the distribution"""

        self.lambtha = float(lambtha)
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1/(sum(data)/len(data)))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

    def pdf(self, x):
        """Calculates the PDF for a given time period x"""

        if x < 0:
            return 0
        return (self.lambtha * e ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculates the CDF value for a given time period x"""

        if x < 0:
            return 0
        return 1 - (e ** (-self.lambtha * x))
