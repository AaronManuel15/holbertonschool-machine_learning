#!/usr/bin/env python3
"""Task 6: builds, trains, and saves a NN classifier"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """X_train: np.ndarray containing the training input data
    Y_train: np.ndarray containing the training labels
    X_valid: np.ndarray containing the validation input data
    Y_train: np.ndarray containing the validation labels
    layer_sizes: list containing the number of nodes in each layer
    activations: list containing the activation functions of each layer
    alpha: the learning rate of the NN
    iterations: the number of iterations to train over
    save_path: designates where to save the model"""

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(iterations + 1):
        if i % 100 == 0 or i == iterations:
            train_cost, train_acc = sess.run((loss, accuracy),
                                             feed_dict={x: X_train,
                                                        y: Y_train})
            val_cost, val_acc = sess.run((loss, accuracy),
                                         feed_dict={x: X_valid,
                                                    y: Y_valid})
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(val_cost))
            print("\tValidation Accuracy: {}".format(val_acc))
        if i < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
    saver = tf.train.Saver()
    return saver.save(sess, save_path)
