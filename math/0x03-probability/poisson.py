#!/usr/bin/env python3
"""Task 0 for 0x03 - Probability"""
pi = 3.1415926536
e = 2.7182818285


class Poisson():
    """Represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):

        self.lambtha = float(lambtha)
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)/len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

    def factorial(self, number):
        """Performs factorial on number"""

        if number == 0:
            return 1
        for i in range(1, int(number)):
            number = number * i
        return number

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""

        k = int(k)
        if k < 1:
            return 0
        return ((e ** -self.lambtha) * (self.lambtha ** k))/self.factorial(k)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""

        k = int(k)
        if k < 1:
            return 0
        cdf = 0
        lmd = self.lambtha
        for i in range(k + 1):
            cdf += ((e ** -lmd) * (lmd ** i))/self.factorial(i)
        return cdf
