import tensorflow as tf
import numpy as np
from tensorflow import keras

from . import losses


def train(
    model: tf.keras.Model,
    training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    epochs: int,
    n: float,
    patience: int,
    loss_fn: tf.keras.losses.Loss = losses.ssim_loss,
    metrics=["accuracy", losses.ssim_loss],
):

    optimizer = keras.optimizers.Adam(learning_rate=n)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    monitor = "val_loss"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=patience, monitor=monitor, restore_best_weights=True
    )

    the_lr_tr, the_hr_tr, the_lr_vl, the_hr_vl = training_data

    history = model.fit(
        x=the_lr_tr,
        y=the_hr_tr,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(the_lr_vl, the_hr_vl),
        callbacks=[early_stopping],
    )

    return model, history
