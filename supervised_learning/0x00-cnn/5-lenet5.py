#!/usr/bin/env python3
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using Keras
    Args:
        X: K.Input of shape (m, 28, 28, 1) containing the input
            images for the network
            m: number of images
    Returns:
        K.Model compiled to use Adam optimization (with default
            hyperparameters) and accuracy metrics"""

    init = K.initializers.he_normal()
    # convolutional layer 1
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=init)(x)

    # pooling layer 1
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    # convolutional layer 2
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=init)(pool1)

    # pooling layer 2
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)

    # flatten
    flatten = K.layers.Flatten()(pool2)

    # fully connected layer 1
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=init)(flatten)

    # fully connected layer 2
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=init)(fc1)

    # fully connected layer 3 aka prediction layer
    y_pred = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=init)(fc2)

    model = K.models.Model(X, y_pred)

    return model.compile(optimizer=K.optimizers.Adam(),
                         loss='categorical_crossetentropy',
                         metrics=['accuracy'])
