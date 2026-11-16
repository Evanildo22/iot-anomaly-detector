"""
Convolutional Autoencoder (CAE) — encoder and full autoencoder.

Fixed bugs vs original:
  BUG-13  Encoder is exposed as a *separate* Model so the classifier can attach
          attention to the bottleneck embedding, not the decoded output.
  BUG-14  Autoencoder loss changed from binary_crossentropy → mse;
          final decoder activation changed from sigmoid → linear.
          (Input features are StandardScaler-normalised, not binary.)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Conv1D,
    Cropping1D,
    Dropout,
    Input,
    MaxPooling1D,
    UpSampling1D,
)
from tensorflow.keras.optimizers import Adam


def build_encoder(
    input_shape: tuple[int, int],
    filters: int = 64,
    kernel_size: int = 3,
    dropout_rate: float = 0.3,
) -> tuple[Model, tf.Tensor, tf.Tensor]:
    """
    Build a two-stage Conv1D encoder.

    Returns
    -------
    inputs : Keras tensor  (the Input layer)
    encoded : Keras tensor (the bottleneck tensor)
    encoder_model : keras.Model mapping inputs → encoded
    """
    inputs = Input(shape=input_shape, name="encoder_input")
    x = Conv1D(filters, kernel_size, activation="relu",
               padding="same", name="enc_conv1")(inputs)
    x = MaxPooling1D(pool_size=2, padding="same", name="enc_pool1")(x)
    x = Dropout(dropout_rate, name="enc_drop1")(x)
    x = Conv1D(filters // 2, kernel_size, activation="relu",
               padding="same", name="enc_conv2")(x)
    encoded = MaxPooling1D(pool_size=2, padding="same", name="bottleneck")(x)

    encoder_model = Model(inputs, encoded, name="encoder")
    return encoder_model, inputs, encoded


def build_autoencoder(
    input_shape: tuple[int, int],
    filters: int = 64,
    kernel_size: int = 3,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> tuple[Model, Model]:
    """
    Build a full convolutional autoencoder + a separate encoder model.

    FIX BUG-14: reconstruction loss = mse; final activation = linear.
    FIX BUG-13: encoder model is returned separately so the classifier
                can operate on the bottleneck, not the reconstruction.

    Returns
    -------
    autoencoder : keras.Model  (trained end-to-end for reconstruction)
    encoder     : keras.Model  (encoder half only — used by classifier)
    """
    encoder_model, inputs, encoded = build_encoder(
        input_shape, filters, kernel_size, dropout_rate
    )

    # Decoder
    x = Dropout(dropout_rate, name="dec_drop1")(encoded)
    x = Conv1D(filters // 2, kernel_size, activation="relu",
               padding="same", name="dec_conv1")(x)
    x = UpSampling1D(size=2, name="dec_up1")(x)
    x = Conv1D(filters, kernel_size, activation="relu",
               padding="same", name="dec_conv2")(x)
    x = UpSampling1D(size=2, name="dec_up2")(x)
    # FIX BUG-23: the two same-padded pool/upsample stages emit
    # 4*ceil(ceil(L/2)/2) time-steps, which exceeds L unless L % 4 == 0 —
    # crop the excess so reconstruction always matches the input length.
    target_len = input_shape[0]
    up_len = int(x.shape[1])
    if up_len > target_len:
        x = Cropping1D((0, up_len - target_len), name="dec_crop")(x)
    # FIX BUG-14: linear activation for continuous normalised output
    decoded = Conv1D(
        input_shape[-1], kernel_size,
        activation="linear", padding="same", name="reconstruction"
    )(x)

    autoencoder = Model(inputs, decoded, name="autoencoder")
    # FIX BUG-14: mean-squared-error reconstruction loss
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder, encoder_model
