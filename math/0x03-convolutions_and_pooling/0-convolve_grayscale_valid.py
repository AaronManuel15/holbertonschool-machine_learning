#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images
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
    Returns:
        numpy.ndarray: containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output[:, i, j] = np.sum(images[:, i:i + kh, j:j + kw] * kernel,
                                     axis=(1, 2))
    return output
