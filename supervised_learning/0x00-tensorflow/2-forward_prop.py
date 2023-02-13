#!/usr/bin/env python3
"""Task 2: creates the forward prop graph for the NN"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """x: placeholder for input data
    layer_sizes: list containing the number of nodes in each layer
    activations: list containing the activation functions in each layer
    returns: the prediction of the network in tensor form"""

    for layer in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[layer], activations[layer])
    return x
