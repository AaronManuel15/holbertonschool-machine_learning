#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a nn
    Args:
        dZ (numpy.ndarray): shape (m, h_new, w_new, c_new) containing the
            gradient of the output
            m: number of examples
            h_new: height of the output
            w_new: width of the output
            c_new: number of channels in the output
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
        padding (str): 'same' or 'valid'
        stride (tuple): (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image
        Returns:
            tuple: (dA_prev, dW, db)"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0

    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_prev = np.zeros_like(padded)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    for pic in range(0, m):
        for h in range(0, h_new):
            for w in range(0, w_new):
                for c in range(0, c_new):
                    dA_prev[pic, sh*h:sh*h+kh, sw*w:sw*w+kw, :] += \
                        dZ[pic, h, w, c] * W[:, :, :, c]
                    dW[:, :, :, c] += dZ[pic, h, w, c] * \
                        padded[pic, sh*h:sh*h+kh, sw*w:sw*w+kw, :]
                    db[:, :, :, c] += dZ[pic, h, w, c]
    dA_prev = dA_prev[:, ph:ph+h_prev, pw:pw+w_prev, :]
    return dA_prev, dW, db
