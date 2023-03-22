#!/usr/bin/env python3
"""Task 3"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in Deep Residual Learning for
        Image Recognition (2015)
    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12, respectively:
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution and the 1x1
                convolution in the shortcut connection
        s: stride of the first convolution in both the main path and the
            shortcut connection
    Returns:
        activated output of the projection block"""

    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=s,
                            padding='same', kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                            kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                            kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)
    conv4 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=s,
                            padding='same', kernel_initializer=init)(A_prev)
    bn4 = K.layers.BatchNormalization(axis=3)(conv4)
    add = K.layers.Add()([bn3, bn4])
    act3 = K.layers.Activation('relu')(add)
    return act3
