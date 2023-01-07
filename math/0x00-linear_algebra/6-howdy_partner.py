#!/usr/bin/env python3
""" Task 6"""


def cat_arrays(arr1, arr2):
    """ Concatenates two arrays"""

    new_arr = []
    for i in arr1:
        new_arr.append(i)
    for i in arr2:
        new_arr.append(i)

    return new_arr
