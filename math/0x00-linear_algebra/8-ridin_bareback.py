#!/usr/bin/env python3
""" Task 8"""


def mat_mul(mat1, mat2):
    """ Multiplies mat1 and mat2"""

    if len(mat1[0]) != len(mat2):
        return None

    matM = [[sum(a*b for a, b in zip(row, col)) for col in zip(*mat2)]
            for row in mat1]

    return matM
