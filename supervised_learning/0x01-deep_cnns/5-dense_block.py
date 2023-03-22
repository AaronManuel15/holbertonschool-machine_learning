#!/usr/bin/env python3
"""Task 5"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected Convolutional
        Networks
    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block
    Returns:
        The concatenated output of each layer within the Dense Block and the
            number of filters within the concatenated outputs, respectively"""

    init = K.initializers.he_normal()
    for _ in range(layers):
        bn1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(bn1)
        # bottleneck layer to reduce the number of input freature maps
        conv1 = K.layers.Conv2D(filters=4 * growth_rate, kernel_size=(1, 1),
                                padding='same', kernel_initializer=init)(act1)
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        # convolution layer
        conv2 = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                                padding='same', kernel_initializer=init)(act2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
