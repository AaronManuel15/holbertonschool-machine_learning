#!/usr/bin/env python3
""" Task 7"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two 2D matrices along a specific axis"""

    matC = []
    if axis == 0:
        """ Concatenates along rows"""
        try:
            for row in mat1:
                matC.append(row.copy())
            for row in mat2:
                matC.append(row.copy())
            return matC

        except Exception:
            return None

    elif axis == 1:
        """ Concatenates along columns"""
        i = 0
        try:
            for row in mat1:
                matC.append(row.copy())
            for row in matC:
                row.append(*mat2[i])
                i += 1
            return matC

        except Exception:
            return None

    return None
