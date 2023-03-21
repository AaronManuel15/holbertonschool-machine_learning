#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """builds an inception block as described in Going Deeper with
        Convolutions (2014)
    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP:
            F1: number of filters in the 1x1 convolution
            F3R: number of filters in the 1x1 convolution before the
                3x3 convolution
            F3: number of filters in the 3x3 convolution
            F5R: number of filters in the 1x1 convolution before the
                5x5 convolution
            F5: number of filters in the 5x5 convolution
            FPP: number of filters in the 1x1 convolution after the
                max pooling
    Returns:
        concatenated output of the inception block"""

    f1, f3r, f3, f5r, f5, fpp = filters

    conv1 = K.layers.Conv2D(filters=f1, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)
    conv2a = K.layers.Conv2D(filters=f3r, kernel_size=(1, 1), padding='same',
                             activation='relu')(A_prev)
    conv2b = K.layers.Conv2D(filters=f3, kernel_size=(3, 3), padding='same',
                             activation='relu')(conv2a)
    conv3a = K.layers.Conv2D(filters=f5r, kernel_size=(1, 1), padding='same',
                             activation='relu')(A_prev)
    conv3b = K.layers.Conv2D(filters=f5, kernel_size=(5, 5), padding='same',
                             activation='relu')(conv3a)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                  padding='same')(A_prev)
    conv4 = K.layers.Conv2D(filters=fpp, kernel_size=(1, 1), padding='same',
                            activation='relu')(pool1)
    output = K.layers.Concatenate()([conv1, conv2b, conv3b, conv4])
    return output
