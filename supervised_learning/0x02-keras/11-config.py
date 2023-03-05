#!/usr/bin/env python3
"""Task 11"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves a models configuration in JSON format
    Args:
        network: model whose configuration should be saved
        filename: path of the file that the configuration should be saved to
    Returns:
        None"""
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """loads a models configuration in JSON format
    Args:
        filename: path of the file containing the models configuration in JSON
        format
    Returns:
        the loaded model"""
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
