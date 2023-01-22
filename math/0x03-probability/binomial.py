#!/usr/bin/env python3
"""Task 10 - 12 for 0x03 - Probability"""
pi = 3.1415926536
e = 2.7182818285


class Binomial():
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the distribution"""

        self.n = int(n)
        self.p = float(p)
        self.data = data
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            m = sum(data)/len(data)
            v = sum([(result - m)**2 for result in data])/len(data)
            self.p = 1 - (v / m)
            self.n = round((sum(data)/self.p)/len(data))
            self.p = float(m/self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
