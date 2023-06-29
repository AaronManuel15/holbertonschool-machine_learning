#!/usr/bin/env python3
"""Task 2: Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder
    Args:
        input_dims (tuple): Contains the dimensions of the model input
        filters (list): Contains the number of filters for each convolutional
                        layer in the encoder, respectively
        latent_dims (tuple): Contains the dimensions of the latent space
                             representation
    Returns:
        encoder, decoder, auto (tuple): Contains the encoder, decoder and
                                        autoencoder models, respectively
    """

    # Encoder
    inputs = keras.Input(shape=input_dims)
    encoded = inputs

    for f in filters:
        encoded = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    encoder = keras.models.Model(inputs, encoded)

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    decoded = latent_inputs

    for i in range(len(filters) - 2, -1, -1):
        decoded = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                      padding='same')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(filters[-2], (3, 3),
                                  activation='relu',
                                  padding='valid')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                                  padding='same')(decoded)

    decoder = keras.models.Model(latent_inputs, decoded)

    # Autoencoder
    auto = keras.models.Model(inputs, decoder(encoder(inputs)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
