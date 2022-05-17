import tensorflow as tf

from . import helpers


def upscale(model: tf.keras.Model, ds: tf.data.Dataset):
    upscaled = []
    for img in ds:
        img_temp = helpers.norm(img)
        predictions = model.predict(img_temp)
        x = [helpers.denorm(y) for y in predictions[0]]
        upscaled.append(x)
    return upscaled
