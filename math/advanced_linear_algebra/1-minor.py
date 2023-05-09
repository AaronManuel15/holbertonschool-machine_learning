#!/usr/bin/env python3
"""Task 1: Minor"""


def get_matrix_minor(m, i, j):
    """gets the minor of a matrix"""
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]


def determinant(matrix):
    """Calculates the determinant of a matrix
    Args:
        matrix (list): matrix to calculate"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in range(len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        new_m = [[row[n] for n in range(len(matrix)) if n != i]
                 for row in rows]
        det += k * (-1) ** i * determinant(new_m)
    return det


def minor(matrix):
    """Calculates the minor matrix of a matrix
    Args:
        matrix (list): matrix to calculate"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in range(len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    return [[determinant(get_matrix_minor(matrix, i, j))
             for j in range(len(matrix[0]))]
            for i in range(len(matrix))]
