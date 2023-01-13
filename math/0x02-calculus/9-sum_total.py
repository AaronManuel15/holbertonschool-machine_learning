#!/usr/bin/env python3
""" Task 9 for 0x02-Calculus"""
import numpy as np


def summation_i_squared(n):
    """ Calculates the summation of i^2 from 1 to n"""

    if type(n) is not int:
        return None

    array = np.arange(1, n + 1, 1, dtype=int)
    array = square(array)

    return array.sum()


def square(val):
    """Had to do this for pycode"""
    return val ** 2
