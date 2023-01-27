import tensorflow as tf
import numpy as np
from tensorflow import keras

from .train import train
from . import losses


def get_real_discriminator_data(
    training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
):
    lr_train, hr_train, lr_val, hr_val = training_data

    train_length = np.shape(hr_train)[0]
    val_length = np.shape(hr_val)[0]

    real_train_x = hr_train
    real_train_y = np.array([1 for i in range(train_length)])

    real_valid_x = hr_val
    real_valid_y = np.array([1 for i in range(val_length)])

    return real_train_x, real_train_y, real_valid_x, real_valid_y


def get_discriminator_data(
    generator: tf.keras.Model,
    training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    real_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
):
    lr_train, hr_train, lr_val, hr_val = training_data

    train_length = np.shape(lr_train)[0]
    val_length = np.shape(lr_val)[0]

    # fake
    fake_train_x = np.array(generator.predict(lr_train, verbose=0))
    fake_train_y = np.array([0 for i in range(train_length)])

    fake_valid_x = np.array(generator.predict(lr_val, verbose=0))
    fake_valid_y = np.array([0 for i in range(val_length)])

    real_train_x, real_train_y, real_valid_x, real_valid_y = real_data

    train_x = np.concatenate((fake_train_x, real_train_x))
    train_y = np.concatenate((fake_train_y, real_train_y))
    valid_x = np.concatenate((fake_valid_x, real_valid_x))
    valid_y = np.concatenate((fake_valid_y, real_valid_y))

    return train_x, train_y, valid_x, valid_y


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
    generator_loss_fn: tf.keras.losses.Loss,
    generator_batch_size: int,
    discriminator_batch_size: int,
    generator_n: int,
    discriminator_n: float,
    generator_epochs: int = 1,
    discriminator_epochs: int = 1,
    pre_train_epochs: int = None,
    pre_train_batch_size: int = None,
    pre_train: bool = False,
    pre_train_loss_fn: tf.keras.losses.Loss = keras.losses.MeanSquaredError(),
    discriminator_loss_fn: tf.keras.losses.Loss = keras.losses.MeanSquaredError(),
    epoch_start_callback=None,
    gen_history=None,
    dis_history=None,
):

    if gen_history == None:
        gen_history = GanHistory()
    if dis_history == None:
        dis_history = GanHistory()

    if pre_train:
        temp = train(
            generator,
            training_data,
            epochs=pre_train_epochs,
            loss_fn=pre_train_loss_fn,
            batch_size=pre_train_batch_size,
            n=generator_n,
        )
        gen_history.extend(temp)

    real_data = get_real_discriminator_data(training_data)
    lr_tr, hr_tr, lr_vl, hr_vl = training_data

    dis_optimizer = keras.optimizers.Adam(learning_rate=discriminator_n)
    discriminator.compile(
        optimizer=dis_optimizer, loss=discriminator_loss_fn, metrics=["accuracy"]
    )

    gen_optimizer = keras.optimizers.Adam(learning_rate=generator_n)

    generator.compile(
        optimizer=gen_optimizer,
        loss=generator_loss_fn,
        metrics=["accuracy", losses.ssim_loss, generator_loss_fn],
    )

    for i in range(epochs):
        if epoch_start_callback:
            epoch_start_callback(generator, i)

        print(f"GAN Epoch {i+1}/{epochs}")

        train_x, train_y, valid_x, valid_y = get_discriminator_data(
            generator, training_data, real_data
        )

        temp = discriminator.fit(
            x=train_x,
            y=train_y,
            batch_size=discriminator_batch_size,
            epochs=i * discriminator_epochs + discriminator_epochs,
            initial_epoch=i * discriminator_epochs,
            validation_data=(valid_x, valid_y),
        )

        dis_history.extend(temp)

        temp = generator.fit(
            x=lr_tr,
            y=hr_tr,
            batch_size=generator_batch_size,
            epochs=i * generator_epochs + generator_epochs,
            initial_epoch=i * generator_epochs,
            validation_data=(lr_vl, hr_vl),
        )

        gen_history.extend(temp)

    return gen_history, dis_history
