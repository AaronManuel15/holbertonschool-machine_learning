#!/usr/bin/env python3
"""Task 9"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model.
    Args:
        network: is the model to save
        filename: path of the file that the model should be saved to
    Returns: None"""
    network.save(filename)


def load_model(filename):
    """Loads an entire model.
    Args:
        filename: path of the file that the model should be loaded from
    Returns: the loaded model"""
    return K.models.load_model(filename)
