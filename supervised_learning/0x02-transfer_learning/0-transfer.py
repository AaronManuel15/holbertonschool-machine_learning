#!/usr/bin/env python3
"""Task 0: Transfer Knowledge"""
import tensorflow.keras as K
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()


def preprocess_data(X, Y):
    """Pre-processes the data for the model
    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
            where m is the number of data points
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns:
        X_p: preprocessed X
        Y_p: preprocessed Y
    """
    X_p = K.applications.efficientnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def main():
    """Main Function"""
    MODEL_PATH = 'cifar10.h5'

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(32, 32, 3))
    upscale = K.layers.experimental.preprocessing.Resizing(244, 224)(inputs)
    base_model = K.applications.EfficientNetV2S(include_top=False,
                                                weights='imagenet',
                                                input_tensor=upscale,
                                                input_shape=(244, 224, 3))
    base_model.trainable = False
    out = base_model.output
    out = K.layers.Flatten()(out)
    out = K.layers.Dense(10, activation='softmax')(out)
    fullmodel = K.models.Model(inputs=inputs, outputs=out)
    fullmodel.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    fullmodel.fit(X_train, Y_train, batch_size=64, epochs=3,
                  validation_data=(X_test, Y_test))
    fullmodel.save(MODEL_PATH)


if __name__ == '__main__':
    main()
