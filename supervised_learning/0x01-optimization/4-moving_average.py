#!/usr/bin/env python3
"""Task 4"""
import numpy as np


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set
    data: list of data to calculate the moving average of
    beta: the weight used for the moving average

    Returns: a list containing the moving averages of data"""

    weight = 0
    for i in range(len(data)):
        weight = beta*weight + (1 - beta)*data[i]
        data[i] = weight / (1 - beta**(i + 1))
    return data
