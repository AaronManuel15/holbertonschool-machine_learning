#!/usr/bin/env python3
""" Task 9 for 0x02-Calculus"""


def summation_i_squared(n):
    """ Calculates the summation of i^2 from 1 to n"""

    if type(n) is not int:
        return None

    if n > 1:
        return ((n**2) + summation_i_squared(n - 1))
    return n**2
