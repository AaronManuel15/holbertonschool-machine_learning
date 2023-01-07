#!/usr/bin/env python3
""" Task 7"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two 2D matrices along a specific axis"""

    matC = []
    if axis == 0:
        """ Concatenates along rows"""
        if len(mat1[0]) != len(mat2[0]):
            return None
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
        if len(mat1) != len(mat2):
            return None
        try:
            for i in range(len(mat1)):
                matC.append(mat1[i].copy() + mat2[i].copy())
            return matC

        except Exception:
            return None

    return None
