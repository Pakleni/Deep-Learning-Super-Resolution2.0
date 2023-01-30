import tensorflow as tf
import numpy as np
from tensorflow import keras

from .helpers import History


def train(
    model: tf.keras.Model,
    training_data: tuple[tf.data.Dataset, tf.data.Dataset],
    epochs: int,
    n: float,
    loss_fn: tf.keras.losses.Loss,
    patience: int | None = None,
    metrics=["accuracy"],
    history: History | None = None,
):

    optimizer = keras.optimizers.Adam(learning_rate=n)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    monitor = "val_loss"

    callbacks = []
    if patience != None:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=patience, monitor=monitor, restore_best_weights=True
        )
        callbacks.append(early_stopping)

    training, validation = training_data

    temp = model.fit(
        x=training,
        epochs=epochs,
        validation_data=validation,
        callbacks=callbacks,
    )

    if history:
        history.extend(temp)
