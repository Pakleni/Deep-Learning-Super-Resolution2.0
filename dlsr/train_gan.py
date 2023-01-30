import tensorflow as tf
import numpy as np
from tensorflow import keras

from .helpers import History


def train_gan(
    srgan: tf.keras.Model,
    training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    generator_loss_fn: tf.keras.losses.Loss,
    generator_n: int,
    discriminator_loss_fn: tf.keras.losses.Loss,
    discriminator_n: float,
    history: History,
):
    training, validation = training_data

    srgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=discriminator_n),
        g_optimizer=keras.optimizers.Adam(learning_rate=generator_n),
        d_loss=discriminator_loss_fn,
        g_loss=generator_loss_fn,
    )

    history.extend(
        srgan.fit(
            x=training,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation,
        )
    )
