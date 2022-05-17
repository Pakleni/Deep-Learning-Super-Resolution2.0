import tensorflow as tf
import numpy as np
from tensorflow import keras

from .train import train
from . import losses


def get_discriminator_data(
    generator: tf.keras.Model,
    training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
):
    lr_train, hr_train, lr_val, hr_val = training_data

    train_x = []
    train_y = []

    valid_x = []
    valid_y = []

    for x, y in zip(lr_train, hr_train):
        train_x.extend(generator(np.array([x])))
        train_y.extend([1])
        train_x.extend([y])
        train_y.extend([0])

    for x, y in zip(lr_val, hr_val):
        valid_x.extend(generator(np.array([x])))
        valid_y.extend([1])
        valid_x.extend([y])
        valid_y.extend([0])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)

    return train_x, train_y, valid_x, valid_y


def train_discriminator(
    discriminator: tf.keras.Model,
    training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    patience: int,
    n: float,
    loss_fn: tf.keras.losses.Loss = keras.losses.MeanSquaredError(),
):

    train_x, train_y, valid_x, valid_y = training_data

    optimizer = keras.optimizers.Adam(learning_rate=n)

    discriminator.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=patience, restore_best_weights=True
    )

    history = discriminator.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(valid_x, valid_y),
        callbacks=[early_stopping],
    )

    return discriminator, history


class GanHistory:
    def __init__(self):
        self.history = {"loss": [], "val_loss": []}

    def extend(self, x: tf.keras.callbacks.History):
        self.history["val_loss"].extend(x.history["val_loss"])
        self.history["loss"].extend(x.history["loss"])


def train_gan(
    discriminator: tf.keras.Model,
    generator: tf.keras.Model,
    training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epochs: int,
    generator_epochs: int,
    discriminator_epochs: int,
    generator_loss_fn: tf.keras.losses.Loss,
    generator_batch_size: int,
    discriminator_batch_size: int,
    generator_patience: int,
    generator_n: int,
    discriminator_patience: int,
    discriminator_n: float,
    pre_train_epochs: int,
    pre_train: bool = True,
    pre_train_loss_fn: tf.keras.losses.Loss = losses.ssim_loss,
    discriminator_loss_fn: tf.keras.losses.Loss = keras.losses.MeanSquaredError(),
    epoch_start_callback=None,
):

    gen_history = GanHistory()
    dis_history = GanHistory()

    if pre_train:
        generator, temp = train(
            generator,
            training_data,
            epochs=pre_train_epochs,
            loss_fn=pre_train_loss_fn,
            batch_size=generator_batch_size,
            patience=generator_patience,
            n=generator_n,
        )
        gen_history.extend(temp)

    for i in range(epochs):
        if epoch_start_callback:
            epoch_start_callback(generator, i)

        descriminator_data = get_discriminator_data(generator, training_data)
        discriminator, temp = train_discriminator(
            discriminator=discriminator,
            training_data=descriminator_data,
            epochs=discriminator_epochs,
            loss_fn=discriminator_loss_fn,
            batch_size=discriminator_batch_size,
            patience=discriminator_patience,
            n=discriminator_n,
        )
        dis_history.extend(temp)
        generator, temp = train(
            generator,
            training_data,
            epochs=generator_epochs,
            loss_fn=generator_loss_fn,
            metrics=["accuracy", losses.ssim_loss, generator_loss_fn],
            batch_size=generator_batch_size,
            patience=generator_patience,
            n=generator_n,
        )
        gen_history.extend(temp)

    return generator, gen_history, discriminator, dis_history
