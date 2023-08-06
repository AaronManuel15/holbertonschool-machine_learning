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

    # Count appearances in references for each unigram
    # This didn't actually happen. Worked very well until an edge case I could
    # not understand.
    ref_app = []
    for ref in references:
        for word in unigrams:
            if word in ref and word not in ref_app:
                ref_app.append(word)

    # Calculate Precision
    uni_len = len(ref_app)
    sent_len = len(sentence)
    P = uni_len / sent_len

    # Calculate Brevity Penalty
    BP = 1
    closest_length = min(len(ref) for ref in references)
    if sent_len < closest_length:
        BP = np.exp(1 - (closest_length/sent_len))

    return P * BP
