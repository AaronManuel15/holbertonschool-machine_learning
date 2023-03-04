#!/usr/bin/env python3
"""Task 6"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent
    Args:
        network: is the model to train
        data: is a numpy.ndarray of shape (m, nx) containing the input data
        labels: is a one-hot numpy.ndarray of shape (m, classes) containing the
            labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: is the data to validate the model with, if not None
        early_stopping: is a boolean that indicates whether early stopping
            should be used. Performed in validation_data exists and
            based on validation loss
        patience: is the patience used for early stopping
        verbose: boolean that determines if output should be printed
            during training
        shuffle: boolean that determines whether to shuffle the batches
            every epoch.
            Normally, it is a good idea to shuffle, but for reproducibility,
            we have chosen to set the default to False.
    Returns: the History object generated after training the model"""

    callback = K.callbacks.EarlyStopping(monitor='loss', patience=patience)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle, callbacks=[callback])
