#!/usr/bin/env python3
"""Task 5"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the GD with Momentum Opt Algo
    alpha: the learning rate
    beta1: momentum weight
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    v: the previous first moment of var

    Returns: the updated variable and the new moment, respectively"""

    v = beta1*v + (1 - beta1)*grad
    var = var - alpha*v
    return var, v
