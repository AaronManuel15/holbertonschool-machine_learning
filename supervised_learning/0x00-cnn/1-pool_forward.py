#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs pooling on images
    Args:
        A_prev (numpy.ndarray): shape (m, h, w, c) containing multiple images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
            c: number of channels in the image
        kernel_shape (tuple): (kh, kw) containing the kernel shape
            for the pooling
            kh: height of the kernel
            kw: width of the kernel
        stride (tuple): (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image
        mode (str): indicates the type of pooling
            max: max pooling
            avg: average pooling
        Returns:
            numpy.ndarray: containing the pooled images"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    if mode == 'max':
        calc = np.max
    elif mode == 'avg':
        calc = np.average

    h = (h - kh) // sh + 1
    w = (w - kw) // sw + 1
    output = np.zeros((m, h, w, c))
    for i in range(0, h):
        for j in range(0, w):
            output[:, i, j, :] = calc(A_prev[:, sh*i:sh*i+kh, sw*j:sw*j+kw],
                                      axis=(1, 2))
    return output
