#!/usr/bin/env python3
"""Task 10"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves a models weights
    Args:
        network: model whose weights should be saved
        filename: path of the file that the weights should be saved to
        save_format: format in which the weights should be saved
    Returns:
        None"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads a models weights
    Args:
        network: model to which the weights should be loaded
        filename: path of the file that the weights should be loaded from
    Returns:
        None"""
    network.load_weights(filename)
    return None
