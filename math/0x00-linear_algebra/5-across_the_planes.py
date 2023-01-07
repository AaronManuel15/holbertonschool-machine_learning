#!/usr/bin/env python3
""" Task 5"""


def add_matrices2D(mat1, mat2):
    """ Adds two matrices element-wise"""

    if len(mat1) != len(mat2):
        return None

    try:
        mat = []
        row = []
        for i in range(len(mat1)):
            if len(mat1[i]) != len(mat2[i]):
                return None
            for j in range(len(mat1[i])):
                row.append(mat1[i][j] + mat2[i][j])
            mat.append(row)
            row = []
        return mat

    except Exception:
        return None
