from tensorflow import keras
from keras import layers

from . import custom_layers


def resnet(frames=64):
    Input_img = keras.Input(shape=(None, None, 3))

    x = custom_layers.padded_conv(Input_img, 64, (3, 3), activation="relu")

    x = custom_layers.res(x=x, frames=frames)
    x = custom_layers.res(x=x, frames=frames)
    x = custom_layers.res(x=x, frames=frames)
    x = custom_layers.res(x=x, frames=frames)
    x = custom_layers.res(x=x, frames=frames)
    x = custom_layers.res(x=x, frames=frames)

    x = layers.UpSampling2D(size=(2, 2))(x)

    x = custom_layers.padded_conv(x, frames // 2, (1, 1), activation="relu")

    x = custom_layers.res(x=x, frames=frames // 2)
    x = custom_layers.res(x=x, frames=frames // 2)
    x = custom_layers.res(x=x, frames=frames // 2)
    x = custom_layers.res(x=x, frames=frames // 2)
    x = custom_layers.res(x=x, frames=frames // 2)
    x = custom_layers.res(x=x, frames=frames // 2)

    decoded = custom_layers.padded_conv(x, 3, (3, 3), activation="sigmoid")

    return keras.Model(Input_img, decoded)
