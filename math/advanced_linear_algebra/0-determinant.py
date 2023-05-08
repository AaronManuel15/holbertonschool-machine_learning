#!/usr/bin/env python3
"""Task 0: Determinant"""


def get_mat_minor(m, i, j):
    """Gets the minor of a matrix"""
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]


def determinant(matrix):
    """Calculates the determinant of a matrix
    Args:
        matrix (list): matrix to calculate"""
    if matrix == [[]]:
        return 1
    if (type(matrix) and matrix is not list
        or all(type(row) is not list for row in matrix)):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    det = 0
    for c in range(len(matrix)):
        det += ((-1)**c)*matrix[0][c]*determinant(get_mat_minor(matrix, 0, c))

    return det
