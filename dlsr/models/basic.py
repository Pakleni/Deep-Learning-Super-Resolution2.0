import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def basic():
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    Input_img = keras.Input(shape=(48, 48, 3))  # 48

    x = tf.pad(Input_img, paddings, "SYMMETRIC")  # 98
    x = layers.Conv2D(512, (3, 3), activation="relu")(x)  # 96

    x = tf.pad(x, paddings, "SYMMETRIC")  # 98
    x = layers.Conv2D(512, (3, 3), activation="relu")(x)  # 96

    x = tf.pad(x, paddings, "SYMMETRIC")  # 98
    x = layers.Conv2D(512, (3, 3), activation="relu")(x)  # 96

    x = layers.Conv2D(256, (1, 1), activation="relu")(x)  # 96

    x = layers.UpSampling2D(size=(2, 2))(x)  # 96

    x = tf.pad(x, paddings, "SYMMETRIC")  # 98
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)  # 96

    x = tf.pad(x, paddings, "SYMMETRIC")  # 98
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)  # 96

    x = tf.pad(x, paddings, "SYMMETRIC")  # 98
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)  # 96

    x = tf.pad(x, paddings, "SYMMETRIC")  # 98
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid")(x)  # 96

    # model done
    model = keras.Model(Input_img, decoded)
    return model
