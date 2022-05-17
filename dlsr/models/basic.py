from tensorflow import keras
from keras import layers

from . import custom_layers


def basic(frames=128):

    Input_img = keras.Input(shape=(None, None, 3))

    x = custom_layers.basic_cluster(Input_img, frames)

    x = layers.UpSampling2D(size=(2, 2))(x)

    x = custom_layers.basic_cluster(x, frames // 2)

    decoded = custom_layers.padded_conv(x, 3, (3, 3), "sigmoid")

    return keras.Model(Input_img, decoded)
