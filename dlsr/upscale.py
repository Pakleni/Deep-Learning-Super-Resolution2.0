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


if __name__ == "__main__":
    from .tests import (
        test_upscale_from_folder,
        test_upscale_with_validation_set,
        test_gan_upscale_from_folder,
    )

    test_gan_upscale_from_folder()
