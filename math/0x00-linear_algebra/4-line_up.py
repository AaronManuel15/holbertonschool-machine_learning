#!/usr/bin/env python3
""" Task 4"""


def add_arrays(arr1, arr2):
    """ Adds two arrays element-wise"""

    try:
        arr = []
        for i in range(len(arr1)):
            arr.append(arr1[i] + arr2[i])
        return arr

    except Exception:
        return None
