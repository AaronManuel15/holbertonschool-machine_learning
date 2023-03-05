#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a same convolution on grayscale images
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
    Returns:
        numpy.ndarray: containing the convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    h = h + 2 * ph - kh + 1
    w = w + 2 * pw - kw + 1
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(images_padded[:, i:i + kh, j:j + kw] *
                                     kernel, axis=(1, 2))
    return output
