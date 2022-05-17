import tensorflow as tf
from tensorflow import keras
from keras import layers

from . import custom_layers


def discriminator(frames=128, image_size=96):
    Input_img = keras.Input(shape=(image_size, image_size, 3))

    x = Input_img
    for i in range(6):
        x = custom_layers.padded_conv(x, frames, (3, 3), activation="relu")

    x = layers.LeakyReLU()(x)

    for i in range(4):
        x = custom_layers.padded_conv(x, frames // 2, (3, 3), activation="relu")

    x = layers.LeakyReLU()(x)

    for i in range(2):
        x = custom_layers.padded_conv(x, frames // 4, (3, 3), activation="relu")

    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    decoded = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(Input_img, decoded)
    return model
