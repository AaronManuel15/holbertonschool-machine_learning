#!/usr/bin/env python3
"""Task 4"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in Deep Residual
        Learning for Image Recognition (2015)
    Returns:
        a Keras model"""

    init = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                            padding='same', kernel_initializer=init)(inputs)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                  padding='same')(act1)
    block1 = projection_block(pool1, [64, 64, 256], s=1)
    block2 = identity_block(block1, [64, 64, 256])
    block3 = identity_block(block2, [64, 64, 256])
    block4 = projection_block(block3, [128, 128, 512])
    block5 = identity_block(block4, [128, 128, 512])
    block6 = identity_block(block5, [128, 128, 512])
    block7 = identity_block(block6, [128, 128, 512])
    block8 = projection_block(block7, [256, 256, 1024])
    block9 = identity_block(block8, [256, 256, 1024])
    block10 = identity_block(block9, [256, 256, 1024])
    block11 = identity_block(block10, [256, 256, 1024])
    block12 = identity_block(block11, [256, 256, 1024])
    block13 = identity_block(block12, [256, 256, 1024])
    block14 = projection_block(block13, [512, 512, 2048])
    block15 = identity_block(block14, [512, 512, 2048])
    block16 = identity_block(block15, [512, 512, 2048])
    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7),
                                      padding='valid')(block16)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(pool2)
    return K.models.Model(inputs=inputs, outputs=dense)
