#!/usr/bin/env python3
"""Task 0: Vanilla Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder
    Args:
        input_dims (int): Contains the dimensions of the model input
        hidden_layers (list): Contains the number of nodes for each hidden
            layer; the hidden layers should be reversed for the decoder
        latent_dims (int): Contains the dimensions of the latent space
    Returns:
        encoder, decoder, auto (tuple): Contains the encoder, decoder and
            autoencoder models, respectively"""

    # Initialize the inputs for encoder and decoder
    input_img = keras.Input(shape=(input_dims,))
    decode_img = keras.Input(shape=(latent_dims,))

    # Encoder layers
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input_img)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    # Decoder layers
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(decode_img)
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Create the encoder portion of the model
    encoder = keras.models.Model(input_img, encoded)
    encoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Create the decoder portion of the model
    decoder = keras.models.Model(decode_img, decoded)
    decoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Create the autoencoder model by combining the encoder & decoder portions
    autoEncoder = keras.models.Model(input_img, decoder(encoder(input_img)))
    autoEncoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoEncoder
