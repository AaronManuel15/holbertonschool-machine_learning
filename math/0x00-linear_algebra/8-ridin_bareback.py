#!/usr/bin/env python3
""" Task 8"""


def mat_mul(mat1, mat2):
    """ Multiplies mat1 and mat2"""

    if len(mat1[0]) != len(mat2):
        return None

    m = zip(*mat2)
    matM = [[sum(a*b for a, b in zip(r, c)) for c in m] for r in mat1]

    return matM
