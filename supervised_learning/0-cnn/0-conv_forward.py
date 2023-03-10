#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional layer of a neural
    Args:
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        W (numpy.ndarray): shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution
            kh: height of the kernel
            kw: width of the kernel
            c_prev: number of channels in the previous layer
            c_new: number of channels in the output
        b (numpy.ndarray): shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        activation (function): activation function applied to the convolution
        padding (str): 'same' or 'valid'
        stride (tuple): (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image"""

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0

    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    h = (h_prev + 2 * ph - kh) // sh + 1
    w = (w_prev + 2 * pw - kw) // sw + 1
    output = np.zeros((m, h, w, c_new))
    for k in range(0, c_new):
        for i in range(0, h):
            for j in range(0, w):
                output[:, i, j, k] = np.sum(padded[:,
                                                   sh*i:sh*i+kh,
                                                   sw*j:sw*j+kw] *
                                            W[:, :, :, k],
                                            axis=(1, 2, 3))
    return activation(output + b)
