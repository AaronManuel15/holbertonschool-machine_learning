#!/usr/bin/env python3
"""Task 0 for 0x03 - Probability"""


class Poisson():
    """Represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):

        self.lambtha = float(lambtha)
        if data:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)/len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
