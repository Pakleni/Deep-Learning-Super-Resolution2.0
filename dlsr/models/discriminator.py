import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def discriminator():
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    Input_img = keras.Input(shape=(96, 96, 3))

    x = Input_img
    for i in range(6):
        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)

    x = layers.LeakyReLU()(x)

    for i in range(4):
        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)

    x = layers.LeakyReLU()(x)

    for i in range(2):
        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(32, (3, 3), activation="relu")(x)

    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    decoded = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(Input_img, decoded)
    return model


if __name__ == "__main__":
    print(discriminator().summary())
