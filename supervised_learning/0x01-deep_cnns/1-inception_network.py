#!/usr/bin/env python3
"""Task 1"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """builds the inception network as described in Going Deeper with
        Convolutions (2014)
    Returns:
        Keras model"""

    init = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same', activation='relu',
                            kernel_initializer=init)(inputs)
    poo1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                 padding='same')(conv1)
    conv2a = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                             activation='relu',
                             kernel_initializer=init)(poo1)
    conv2b = K.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same',
                             activation='relu',
                             kernel_initializer=init)(conv2a)
    pool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(conv2b)
    block1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    block2 = inception_block(block1, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(block2)
    block3 = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    block4 = inception_block(block3, [160, 112, 224, 24, 64, 64])
    block5 = inception_block(block4, [128, 128, 256, 24, 64, 64])
    block6 = inception_block(block5, [112, 144, 288, 32, 64, 64])
    block7 = inception_block(block6, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(block7)
    block8 = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    block9 = inception_block(block8, [384, 192, 384, 48, 128, 128])
    pool5 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7),
                                      padding='valid')(block9)
    dropout = K.layers.Dropout(rate=0.4)(pool5)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(dropout)
    model = K.Model(inputs=inputs, outputs=dense)
    return model
