#!/usr/bin/env python3
""" Task 3"""


def matrix_transpose(matrix):
    """ Calculates the transpose of a matrix"""

    return list(map(list, zip(*matrix)))
