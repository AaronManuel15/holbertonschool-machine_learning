#!/usr/bin/env python3
"""Task 3"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a NN
    Args:
        dA: ndarray of shape (m, h_new, w_new, c_new) containing the partial
            derivatives with respect to the output of the pooling layer
            m: is the number of examples
            h_new: is the height of the output
            w_new: is the width of the output
            c: is the number of channels
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
            the output of the previous layer
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
        kernel_shape: is a tuple of (kh, kw) containing the size of the kernel
            for the pooling
            kh: is the kernel height
            kw: is the kernel width
        stride: is a tuple of (sh, sw) containing the strides for the pooling
            sh: is the stride for the height
            sw: is the stride for the width
        mode: is a string containing either max or avg, indicating whether to
            perform maximum or average pooling, respectively

    Returns: the partial derivatives with respect to the previous layer
        (dA_prev)"""

    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)

    for pic in range(0, m):
        for h in range(0, h_new):
            for w in range(0, w_new):
                for c in range(0, c_new):
                    if mode == 'max':
                        tmp = A_prev[pic, sh*h:sh*h+kh, sw*w:sw*w+kw, c]
                        mask = (tmp == np.max(tmp))
                        dA_prev[pic, sh*h:sh*h+kh, sw*w:sw*w+kw, c] += \
                            mask * dA[pic, h, w, c]
                    elif mode == 'avg':
                        dA_prev[pic, sh*h:sh*h+kh, sw*w:sw*w+kw, c] += \
                            np.ones((kh, kw)) * dA[pic, h, w, c] / (kh * kw)
    return dA_prev
