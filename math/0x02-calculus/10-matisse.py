#!/usr/bin/env python3
""" Task 10 for 0x02-Calculus"""


def poly_derivative(poly):
    """ Calculates the derivative of poly"""

    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]

    for power in range(len(poly)):
        poly[power] = (poly[power] * power)
    poly = poly[1:]
    return poly
