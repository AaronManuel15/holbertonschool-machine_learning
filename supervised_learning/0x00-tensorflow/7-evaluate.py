#!/usr/bin/env python3
"""Task 7: Evaluates the output of a NN from a data set"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """X: numpy.ndarray containing the input data to evaluate
    Y: numpy.ndarray containing the one-hot labels for X
    save_path: is the location to load the model from
    Returns: the network's prediction, accuracy, and loss, respectively"""

    sess = tf.Session()
    saved = tf.train.import_meta_graph(save_path + '.meta')
    saved.restore(sess, save_path)
    graph = tf.get_default_graph()
    x = graph.get_collection("x")[0]
    y = graph.get_collection("y")[0]
    y_pred = graph.get_collection("y_pred")[0]
    accuracy = graph.get_collection("accuracy")[0]
    loss = graph.get_collection("loss")[0]
    return sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})
