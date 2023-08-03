#!/usr/bin/env python3
"""Task 4: FastText"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Creates and trains a gensim fastText model
    Args:
        sentences: list of sentences to be trained on
        size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word for use in training
        window: maximum distance between the current and predicted word
                within a sentence
        negative: size of negative sampling
        cbow: boolean to determine the training type; True is for CBOW;
              False is for Skip-gram
        iterations: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model
    Returns:
        the trained model"""

    sg = 1
    if cbow:
        sg = 0

    model = FastText(sentences, vector_size=size, min_count=min_count,
                     window=window, negative=negative, seed=seed,
                     epochs=iterations, workers=workers, sg=sg)
    return model
