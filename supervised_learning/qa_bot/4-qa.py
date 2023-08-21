#!/usr/bin/env python3
"""Task 4. Answer Questions"""
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to
    answer a question.
    Args:
        question (str): Contains the question to answer.
        reference (str): Contains the reference document from which to find
                         the answer.
    Returns:
        (str): Contains the answer."""

    tkn = BertTokenizer.from_pretrained(('bert-large-uncased-whole'
                                        '-word-masking-finetuned-squad'))
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    q_tokens = tkn.tokenize(question)
    ref_tokens = tkn.tokenize(reference)
    tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + ref_tokens + ['[SEP]']
    input_word_ids = tkn.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = ([0] * (len(q_tokens) + 2) + [1] * (len(ref_tokens) + 1))

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids,
                                                      input_mask,
                                                      input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tkn.convert_tokens_to_string(answer_tokens)

    if not answer:
        return None
    return answer


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


def question_answer_semantic(corpus_path):
    """answers questions from multiple reference texts"""
    while(True):
        question = input('Q: ')
        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print('A: Goodbye')
            break
        else:
            reference = semantic_search(corpus_path, question)
            answer = question_answer(question, reference)
            confused = 'Sorry, I do not understand your question.'
            print('A: {}'.format(answer if answer else confused))
