import tensorflow as tf
import numpy as np
from . import helpers


def upscale(model: tf.keras.Model, ds: tf.data.Dataset, batch_size=None):
    data = ds.map(helpers.normalize_lr)
    preds = model.predict(data, batch_size=batch_size)
    return helpers.denormalize_hr(preds)
