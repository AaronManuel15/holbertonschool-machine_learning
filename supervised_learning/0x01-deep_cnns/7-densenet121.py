#!/usr/bin/env python3
"""Task 7"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in Densely Connected
        Convolutional Networks
    Args:
        growth_rate: growth rate
        compression: compression factor
    Returns:
        the keras model"""

    init = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))
    bn1 = K.layers.BatchNormalization(axis=3)(inputs)
    act1 = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(filters=2 * growth_rate, kernel_size=(7, 7),
                            kernel_initializer=init, padding='same')(act1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(conv1)

    block1, nb_filters = dense_block(pool1, 2 * growth_rate, growth_rate, 6)
    t_layer1, nb_filters = transition_layer(block1, nb_filters, compression)
    block2, nb_filters = dense_block(t_layer1, nb_filters, growth_rate, 12)
    t_layer2, nb_filters = transition_layer(block2, nb_filters, compression)
    block3, nb_filters = dense_block(t_layer2, nb_filters, growth_rate, 24)
    t_layer3, nb_filters = transition_layer(block3, nb_filters, compression)
    block4, nb_filters = dense_block(t_layer3, nb_filters, growth_rate, 16)

    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                      padding='valid')(block4)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(pool2)
    model = K.models.Model(inputs=inputs, outputs=dense)
    return model
