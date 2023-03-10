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

    def z_score(self, x):
        """Calculates the z-score of a given value x"""

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""

        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""

        v = self.stddev
        m = self.mean
        return (1/((2*pi*v**2)**.5))*e**((-(x-m)**2)/(2*v**2))

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""

        return .5*(1 + self.errorfx((x-self.mean)/(self.stddev*2**.5)))

    def errorfx(self, x):
        """Taylor series expansion for error function calc in CDF F(x)"""

        return (2/(pi**.5))*(x-(x**3)/3 + (x**5)/10 - (x**7)/42 + (x**9)/216)
