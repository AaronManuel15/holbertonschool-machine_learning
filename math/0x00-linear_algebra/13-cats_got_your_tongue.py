#!/usr/bin/env python3
""" Task 13"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Concatenates two matrices along specific axis"""

    newMat = np.concatenate((mat1, mat2), axis)

    return newMat
