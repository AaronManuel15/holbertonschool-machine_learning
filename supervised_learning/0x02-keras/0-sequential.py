#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with Keras
    nx: is the number of input features to the network
    layers: list containing the number of nodes in each layer
        of the network
    activations: list containing the activation functions
        used for each layer of the network
    lambtha: is the L2 regularization parameter
    keep_prob: is the probability that a node will be kept for dropout

    Returns: the keras model
    """

    reg = K.regularizers.l2(lambtha)
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=reg,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=reg))
    return model
