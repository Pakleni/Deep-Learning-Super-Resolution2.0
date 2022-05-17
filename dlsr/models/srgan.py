import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def srgan():
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    Input_img = keras.Input(shape=(96, 96, 3))  # 96

    x = layers.Conv2D(64, (3, 3), activation="relu")(Input_img)  # 94
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)  # 92
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)  # 90
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)  # 88

    x = layers.MaxPooling2D((2, 2))(x)  # 44

    x = layers.Conv2D(128, (3, 3), activation="relu")(x)  # 42
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)  # 40

    x = layers.MaxPooling2D((2, 2))(x)  # 20

    x = layers.Conv2D(256, (3, 3), activation="relu")(x)  # 18
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)  # 16
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)  # 14
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)  # 12

    x = layers.MaxPooling2D((2, 2))(x)  # 6

    x = layers.Conv2D(512, (3, 3), activation="relu")(x)  # 4

    x = layers.Flatten()(x)

    decoded = layers.Dense(units=1, activation="sigmoid")(x)  # 1

    # model done
    model = keras.Model(Input_img, decoded)

    return model
