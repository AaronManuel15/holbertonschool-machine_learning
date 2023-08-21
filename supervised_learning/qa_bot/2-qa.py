#!/usr/bin/env python3
"""Task 2. Answer Questions"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


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
    input_type_ids = ([0] * (1 + len(q_tokens)+1) + [1]*(len(ref_tokens) + 1))

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


def answer_loop(reference):
    """Answers questions from a reference text.
    Args:
        reference (str): Contains the reference document from which to find
                         the answer.
    """
    while(True):
        question = input('Q: ')
        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print('A: Goodbye')
            break
        else:
            answer = question_answer(question, reference)
            confused = 'Sorry, I do not understand your question.'
            print('A: {}'.format(answer if answer else confused))
