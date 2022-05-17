import os
import tensorflow as tf
import numpy as np

from .. import *


def test_upscale_from_folder():
    from PIL import Image

    model = tf.keras.models.load_model(
        "./saved-models/tests/model.h5", custom_objects=losses.get_custom_objects()
    )

    ds = helpers.dataset_from_folder("input")

    upscaled = upscale(model, ds)

    for i, x in enumerate(upscaled):
        im = Image.fromarray(np.uint8(x))
        im.save(os.path.join("output", f"converted{i}.png"), format="png")


def test_upscale_with_validation_set():
    from tensorflow.python.data.experimental import AUTOTUNE
    from PIL import Image
    from ..data import DIV2K

    loader = DIV2K(type="valid")
    ds = loader.dataset(batch_size=1, random_transform=False, crop_images=True)

    lr = ds.map(lambda a, b: a)

    model = tf.keras.models.load_model(
        "./saved-models/tests/model.h5", custom_objects=losses.get_custom_objects()
    )

    upscaled = upscale(model, lr.take(10))

    for i, x in enumerate(upscaled):
        im = Image.fromarray(np.uint8(x))
        im.save(os.path.join("output/validation", f"converted{i}.png"), format="png")


def test_gan_upscale_from_folder():
    from PIL import Image

    srgan = tf.keras.models.load_model(
        "./saved-models/tests/gan/discriminator.h5",
        custom_objects=losses.get_custom_objects(),
    )
    model = tf.keras.models.load_model(
        "./saved-models/tests/gan/generator.h5",
        custom_objects=losses.get_custom_objects(srgan),
    )

    ds = helpers.dataset_from_folder("input")

    upscaled = upscale(model, ds)

    for i, x in enumerate(upscaled):
        im = Image.fromarray(np.uint8(x))
        im.save(os.path.join("output/gan", f"converted{i}.png"), format="png")
