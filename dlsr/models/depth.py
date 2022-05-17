import tensorflow as tf
from tensorflow import keras

from . import custom_layers


def depth(frames=128):
    Input_img = keras.Input(shape=(None, None, 3))  # 48

    x = custom_layers.down(Input_img, frames // 2)
    x = custom_layers.down(x, frames)

    x = custom_layers.padded_conv(x, 3 * (2**2), (3, 3), activation="sigmoid")

    decoded = tf.nn.depth_to_space(x, 2, data_format="NHWC", name=None)

    return keras.Model(Input_img, decoded)
