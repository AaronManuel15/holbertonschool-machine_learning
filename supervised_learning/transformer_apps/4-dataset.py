#!/usr/bin/env python3
"""Task 4. Create Masks"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len) -> None:
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_length=self.max_len):
            """filter method"""
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(2**15, reshuffle_each_iteration=True).padded_batch(self.batch_size)
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(filter_max_length).padded_batch(self.batch_size)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        Subword = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = Subword.build_from_corpus((pt.numpy() for pt, _ in data),
                                                 target_vocab_size=2**15)
        tokenizer_en = Subword.build_from_corpus((en.numpy() for _, en in data),
                                                 target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        pt_vsize = self.tokenizer_pt.vocab_size
        en_vsize = self.tokenizer_en.vocab_size
        pt_tokens = [pt_vsize] + self.tokenizer_pt.encode(pt.numpy()) + \
                    [pt_vsize + 1]
        en_tokens = [en_vsize] + self.tokenizer_en.encode(en.numpy()) + \
                    [en_vsize + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """acts as a tensorflow wrapper for the encode instance method"""
        pt_lang, en_lang = tf.py_function(func=self.encode,
                                          inp=[pt, en],
                                          Tout=[tf.int64, tf.int64])
        pt_lang.set_shape([None])
        en_lang.set_shape([None])
        return pt_lang, en_lang
