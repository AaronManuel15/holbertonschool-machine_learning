#!/usr/bin/env python3
"""Task 0: Unigram BLEU score"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence
    Args:
        references: list of reference translations
            each ref translation is a list of the words in the translation
        sentence: list containing the model proposed sentence
    Returns:
        the unigram BLEU score"""

    # Calculating P
    # Remove duplicates from sentence
    seen = set()
    unigrams = [x for x in sentence if not (x in seen or seen.add(x))]

    # Count appearances in references and sentence for each unigram
    sent_app = [sentence.count(i) for i in unigrams]
    ref_app = []
    for ref in references:
        ref_app.append([ref.count(x) for x in unigrams])
    ref_app = np.array(ref_app)

    # Calculate Precision
    uni_len = len(unigrams)
    sent_len = len(sentence)
    P = uni_len / sent_len

    # Calculate Brevity Penalty
    BP = 1
    closest_length = min(len(ref) for ref in references)
    if sent_len < closest_length:
        BP = np.exp(1 - (closest_length/sent_len))

    return P * BP
