#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images
    Args:
        images (numpy.ndarray): shape (m, h, w) containing multiple grayscale
            images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel (numpy.ndarray): shape (kh, kw) containing the kernel for the
            convolution
            kh: height of the kernel
            kw: width of the kernel
        padding (tuple): (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image
            (padded with 0’s)
        stride (tuple): (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image
        Returns:
            numpy.ndarray: containing the convolved images"""

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    h = (h + 2 * ph - kh) // sh + 1
    w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, h, w))
    for i in range(0, h):
        for j in range(0, w):
            output[:, i, j] = np.sum(padded[:, sh*i:sh*i+kh, sw*j:sw*j+kw] *
                                     kernel, axis=(1, 2))
    return output
