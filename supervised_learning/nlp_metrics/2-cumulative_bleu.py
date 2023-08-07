#!/usr/bin/env python3
"""Task 2: Cumulative N-gram BLEU score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence
    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use
    Returns:
        the cumulative n-gram BLEU score"""

    # Calculate the precision for each n-gram size
    precisions = []
    for i in range(1, n + 1):
        precisions.append(np.log(precision(references, sentence, i)))
    P = np.exp(np.sum(precisions) / n)

    # Calculate Brevity Penalty
    BP = 1
    sent_len = len(sentence)
    closest_length = min(len(ref) for ref in references)
    if sent_len < closest_length:
        BP = np.exp(1 - (closest_length/sent_len))
    return P * BP


def n_gram_generator(sentence, n):
    """Generates a list of n-grams from a sentence
    Args:
        sentence: list containing the model proposed sentence
        n: size of the n-gram to generate
    Returns:
        list of n-grams"""

    n_grams = []
    for i in range(len(sentence) - n + 1):
        if sentence[i:i+n] not in n_grams:
            n_grams.append(sentence[i:i+n])

    return n_grams


def n_gram_appearance(n_gram, sentence):
    """Counts the number of appearances of a n-gram in a sentence
    Args:
        n_gram: n-gram to search for
        sentence: list containing the model proposed sentence
    Returns:
        number of appearances of n_gram in sentence"""

    count = 0
    for i in range(len(sentence) - len(n_gram) + 1):
        if sentence[i:i+len(n_gram)] == n_gram:
            count += 1

    return count


def precision(references, sentence, n):
    """Calculates the precision for a sentence and a n-gram value
    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
        n: size of the n-gram to use
    Returns:
        precision score"""

    n_grams = n_gram_generator(sentence, n)

    # Calculate the precision for each n-gram
    # Count appearances in sentence and references for each n-gram
    sent_app = []
    ref_app = [[] for ref in references]
    for n_gram in n_grams:
        sent_app.append(n_gram_appearance(n_gram, sentence))
    for i, ref in enumerate(references):
        for n_gram in n_grams:
            ref_app[i].append(n_gram_appearance(n_gram, ref))

    # Merge the counts of appearances in references for max appearance
    ref_app_max = np.dstack(ref_app).max(axis=2)[0]

    # Calculate Precision
    P = np.sum(ref_app_max) / np.sum(sent_app)

    return P
