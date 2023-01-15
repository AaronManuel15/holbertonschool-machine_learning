#!/usr/bin/env python3
""" Task 17 for 0x02-Calculus"""


def poly_integral(poly, C=0):
    """ Calculates the integral of poly"""

    if type(poly) is not list or type(C) not in [int, float] or len(poly) <= 0:
        return None 

    for power in range(1, len(poly)):
        poly[power] = (poly[power] / (power + 1))
    poly.insert(0, C)

    return poly
