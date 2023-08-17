#!/usr/bin/env python3
"""Task 3: Extract Word2Vec"""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer
    Args:
        model: trained gensim word2vec model
    Returns:
        the trainable keras Embedding"""

    keyed_vectors = model.wv
    # vectors themselves, a 2D numpy array
    weights = keyed_vectors.vectors

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=False,
    )
    return layer