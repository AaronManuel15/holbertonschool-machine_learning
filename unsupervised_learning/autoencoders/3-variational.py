#!/usr/bin/env python3
"""Task 3: Variational Autoencoder"""
import tensorflow.keras as keras
kl = keras.layers
kb = keras.backend


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder
    Args:
        input_dims (int): Contains the dimensions of the model input
        hidden_layers (list): Contains the number of nodes for each hidden
            layer; the hidden layers should be reversed for the decoder
        latent_dims (int): Contains the dimensions of the latent space
    Returns:
        encoder, decoder, auto (tuple): Contains the encoder, decoder and
            autoencoder models, respectively"""


# Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for units in hidden_layers:
        x = kl.Dense(units, activation='relu')(x)

    # Latent space
    latent_mean = kl.Dense(latent_dims, activation=None)(x)
    latent_log_variance = kl.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sampling(args):
        l_mean, latent_log_variance = args
        epsilon = kb.random_normal(shape=(kb.shape(l_mean)[0], latent_dims),
                                   mean=0.0, stddev=1.0)
        return latent_mean + kb.exp(0.5 * latent_log_variance) * epsilon

    latent_space = kl.Lambda(sampling)([latent_mean, latent_log_variance])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for units in reversed(hidden_layers):
        x = kl.Dense(units, activation='relu')(x)

    decoder_outputs = kl.Dense(input_dims, activation='sigmoid')(x)

    # Define encoder and decoder models
    encoder = keras.Model(encoder_inputs,
                          [latent_space, latent_mean, latent_log_variance],
                          name='encoder')
    decoder = keras.Model(decoder_inputs,
                          decoder_outputs,
                          name='decoder')

    # Define full autoencoder model
    autoencoder_outputs = decoder(encoder(encoder_inputs)[0])
    autoencoder = keras.Model(encoder_inputs,
                              autoencoder_outputs,
                              name='autoencoder')

    # Compile the autoencoder model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
