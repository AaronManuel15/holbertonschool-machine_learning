#!/usr/bin/env python3
"""Task 6"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely Connected
        Convolutional Networks
    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        compression: compression factor for the transition layer
    Returns:
        The output of the transition layer and the number of filters within
            the output, respectively"""

    init = K.initializers.he_normal()
    bn1 = K.layers.BatchNormalization(axis=3)(X)
    act1 = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(filters=int(nb_filters * compression),
                            kernel_size=(1, 1), padding='same',
                            kernel_initializer=init)(act1)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding='valid')(conv1)
    return avg_pool, int(nb_filters * compression)
