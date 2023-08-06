#!/usr/bin/env python3
"""Task 0: Unigram BLEU score"""
import numpy as np
import math
np.seterr(divide='ignore', invalid='ignore')


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
    precision = []
    for i, uni in enumerate(sent_app):
        try:
            precision.append(int(uni) / ref_app[..., i].max())
        except RuntimeWarning:
            precision.append(0)
    sent_len = len(sentence)
    precision = np.sum([v for v in precision if not math.isinf(v)]) / sent_len

    # Calculate Brevity Penalty
    bp = 1
    uni_len = len(unigrams)
    closest_length = min(references, key=lambda ref: abs(len(ref) - uni_len))
    if uni_len < len(closest_length):
        bp = np.exp(1 - len(closest_length)/uni_len)

    return precision * bp
