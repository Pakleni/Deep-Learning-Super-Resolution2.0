from tensorflow import keras
from keras import layers

from . import custom_layers


def inception(frames=64):
    Input_img = keras.Input(shape=(None, None, 3))

    x = custom_layers.basic_cluster(Input_img, frames)
    x1 = custom_layers.inception_cluster(x, frames // 2)

    x2 = layers.MaxPooling2D((2, 2))(x1)  # (f/2)

    x3 = custom_layers.inception_cluster(x2, frames // 2)  # (f/2)

    x4 = layers.Concatenate()([x2, x3])  # (f/2 + f/2)

    x5 = layers.UpSampling2D(size=(2, 2))(x4)  # (f)

    x6 = custom_layers.padded_conv(x5, frames // 2, (1, 1), activation="relu")  # (f/2)

    x7 = custom_layers.inception_cluster(x6, frames // 2)  # (f/2)

    x8 = layers.Concatenate()([x, x5, x7])  # (f/2 + f + f/2)

    x9 = custom_layers.padded_conv(x8, frames, (1, 1), activation="relu")  # (f)

    x10 = layers.UpSampling2D(size=(2, 2))(x9)  # (f)

    x11 = custom_layers.basic_cluster(x10, frames)  # (f/2)

    decoded = custom_layers.padded_conv(x11, 3, (3, 3), activation="sigmoid")

    return keras.Model(Input_img, decoded)
