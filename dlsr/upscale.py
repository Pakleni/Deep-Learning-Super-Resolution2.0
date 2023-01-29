import tensorflow as tf
import numpy as np
from . import helpers


def upscale(model: tf.keras.Model, ds: tf.data.Dataset | np.ndarray, batch_size=None):

    # This function only accepts Datasets with 1 batch
    data = np.array([helpers.normalize(img[0]) for img in ds])

    preds = model.predict(data, batch_size=batch_size)

    return [[helpers.denormalize(x) for x in pred] for pred in preds]
