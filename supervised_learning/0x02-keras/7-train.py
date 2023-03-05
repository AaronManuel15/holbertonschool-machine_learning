#!/usr/bin/env python3
"""Task 7"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent, early stopping,
    and learning rate decay.
    Args:
        network: is the model to train
        data: is a numpy.ndarray of shape (m, nx) containing the input data
        labels: is a one-hot numpy.ndarray of shape (m, classes) containing the
            labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: is the data to validate the model with, if not None
        early_stopping: is a boolean that indicates whether early stopping
            should be used. Performed if validation_data exists and
            based on validation loss
        patience: is the patience used for early stopping
        learning_rate_decay: is a boolean that indicates whether learning rate
            should be used. Performed if validation_data exists. Inverse time
            decay. Stepwise fashion after each epoch.
        alpha: is the initial learning rate
        decay_rate: is the decay rate
        verbose: boolean that determines if output should be printed
            during training
        shuffle: boolean that determines whether to shuffle the batches
            every epoch.
            Normally, it is a good idea to shuffle, but for reproducibility,
            we have chosen to set the default to False.
    Returns: the History object generated after training the model"""

    EarlyStop, LRDecay = None, None

    if validation_data and early_stopping:
        EarlyStop = K.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                              patience=patience)
    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """Scheduler"""
            return alpha / (1 + decay_rate * epoch)
        LRDecay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle,
                       callbacks=[EarlyStop, LRDecay])
