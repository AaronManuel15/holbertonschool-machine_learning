#!/usr/bin/env python3
"""Task 3. Semantic Search"""
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


def load_corpus(corpus_path):
    """Loads the corpus into memory"""
    documents = []
    for filename in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, filename), "r",
                  encoding="utf-8") as f:
            content = f.read()
        documents.append(content)
    return documents


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.
    Returns: the reference text of the best-matching document.
    """

    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(model_url)

    documents = load_corpus(corpus_path)
    sentence_embeddings = model([sentence])[0]
    corpus_embeddings = model(documents)
    similarity_scores = cosine_similarity([sentence_embeddings],
                                          corpus_embeddings)[0]
    idx = np.argmax(similarity_scores)
    return documents[idx]
