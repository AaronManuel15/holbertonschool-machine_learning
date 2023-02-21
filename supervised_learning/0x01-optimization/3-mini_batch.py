#!/usr/bin/env python3
"""Task 3"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch descent
    X_train: a numpy.ndarray of shape (m, 784) containing the data
    where m is number of data points and
    784 is the number of features in X
    Y_train: is a one-hot array of shape (m, 10) containing the labels
    10 is the number of classes the model should classify
    X_valid: numpy array of shape (m, 784) containing validation labels
    Y_valid: is a one-hot array of shape (m, 10) containing valid. labels
    batch_size: number of data points in a batch
    epochs: number of times the training should pass through the whole data set
    load_path: path to load the model from
    save_path: path to where the model will be saved after training
    Returns: save_path"""

    sess = tf.Session()
    saved = tf.train.import_meta_graph(load_path + '.meta')
    saved.restore(sess, load_path)
    graph = tf.get_default_graph()
    x = graph.get_collection("x")[0]
    y = graph.get_collection("y")[0]
    accuracy = graph.get_collection("accuracy")[0]
    loss = graph.get_collection("loss")[0]
    train_op = graph.get_collection("train_op")
    m = X_train.shape[0]
    mini_batches = m // batch_size
    if mini_batches % batch_size != 0:
        mini_batches += 1
    for epoch in range(epochs + 1):
        X_shuff, Y_shuff = shuffle_data(X_train, Y_train)
        train_cost, train_accuracy = sess.run((loss, accuracy),
                                              feed_dict={x: X_shuff,
                                                         y: Y_shuff})
        valid_cost, valid_accuracy = sess.run((loss, accuracy),
                                              feed_dict={x: X_valid,
                                                         y: Y_valid})
        print("After {} epochs:".format(epoch))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        if epoch < epochs:
            for mini_batch in range(mini_batches):
                run_feed = {x: X_shuff[mini_batch *
                                       batch_size:batch_size*(mini_batch+1)],
                            y: Y_shuff[mini_batch *
                                       batch_size:batch_size*(mini_batch+1)]}
                sess.run(train_op, feed_dict=run_feed)
                if (mini_batch + 1) % 100 == 0 and mini_batch != 0:
                    mb_cost = sess.run(loss, feed_dict=run_feed)
                    mb_accuracy = sess.run(accuracy, feed_dict=run_feed)
                    print("\tStep {}:".format(mini_batch + 1))
                    print("\t\tCost: {}".format(mb_cost))
                    print("\t\tAccuracy: {}".format(mb_accuracy))
    saver = tf.train.Saver()
    return saver.save(sess, save_path)
