#!/usr/bin/env python3
""" Task 9 for 0x02-Calculus"""


def summation_i_squared(n):
    """ Calculates the summation of i^2 from 1 to n"""

    if n < 1:
        return None

    return sum(map(lambda sq: sq**2, list(range(1, n + 1))))
