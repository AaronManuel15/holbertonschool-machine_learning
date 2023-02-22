#!/usr/bin/env python3
"""Task 9"""
import tensorflow as tf


def update_variables_Adam(alpha, beta1, beta2,
                          epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam opt algo:
    alpha: the learning rate
    beta1: the weight used for the first moment
    beta2: the weight used for the second moment
    epsilon: small number to avoid division by zero
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    v: is the previous first moment of var
    s: is the previous second moment of var
    t: is the time step used for bias correction

    Returns: updated variable, new first moment, new second moment"""

    v = beta1*v + ((1 - beta1)*grad)
    s = beta2*s + ((1 - beta2)*(grad**2))
    vBias = v/(1 - (beta1**t))
    sBias = s/(1 - (beta2**t))
    var = var - alpha*(vBias/((sBias**.5) + epsilon))

    return var, v, s
