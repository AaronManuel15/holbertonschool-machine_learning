#!/usr/bin/env python3
"""Task 6 - 9 for 0x03 - Probability"""
pi = 3.1415926536
e = 2.7182818285


class Normal():
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Defines the normal distribution during initialization"""

        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data)/len(data))
            difSum = 0
            for value in data:
                difSum += (value - self.mean) ** 2
            self.stddev = float((difSum / len(data)) ** .5)
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
