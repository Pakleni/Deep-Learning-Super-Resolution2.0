import tensorflow as tf
from tensorflow import keras
from keras import layers


def block(filters, stride, kernel_size=3):
    def run(x):
        x = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=stride, padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x

    return run


def discriminator(image_size=96):

    Input_img = keras.Input(shape=(image_size, image_size, 3))

    x = Input_img

    x = layers.Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = block(64, 2)(x)
    x = block(128, 1)(x)
    x = block(128, 2)(x)
    x = block(256, 1)(x)
    x = block(256, 2)(x)
    x = block(512, 1)(x)
    x = block(512, 2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    decoded = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(Input_img, decoded)
    return model
