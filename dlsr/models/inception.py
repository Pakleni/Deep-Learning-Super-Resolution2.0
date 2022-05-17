import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def inception():
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    def basic_cluster(x, num):
        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num * 2, (3, 3), activation="relu")(x)

        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num * 2, (3, 3), activation="relu")(x)

        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num * 2, (3, 3), activation="relu")(x)

        x = layers.Conv2D(num, (1, 1), activation="relu")(x)

        return x

    def inception_cluster(x, num):
        x1 = x
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x1_1 = tf.pad(x1, paddings, "SYMMETRIC")
        x1_1 = layers.Conv2D(num / 4, (1, 1), activation="relu")(x1_1)  # 48

        paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        x1_2 = tf.pad(x1, paddings, "SYMMETRIC")
        x1_2 = layers.Conv2D(num / 4, (1, 1), activation="relu")(x1_2)  # 48

        paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        x1_3 = tf.pad(x1, paddings, "SYMMETRIC")
        x1_3 = layers.Conv2D(num / 4, (1, 1), activation="relu")(x1_3)  # 48

        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

        x2_1 = layers.Conv2D(num / 4, (1, 1), activation="relu")(x1)  # 48
        x2_2 = layers.Conv2D(num / 4, (3, 3), activation="relu")(x1_1)  # 48
        x2_3 = layers.Conv2D(num / 4, (5, 5), activation="relu")(x1_2)  # 48
        x2_4 = layers.Conv2D(num / 4, (7, 7), activation="relu")(x1_3)  # 48

        x2 = layers.Concatenate()([x2_1, x2_2, x2_3, x2_4])  # 48 (64)

        return x2

    # input layer
    Input_img = keras.Input(shape=(48, 48, 3))  # 48

    x = basic_cluster(x=Input_img, num=64)  # 48 (64)

    x1 = inception_cluster(x=x, num=64)  # 48 (64)

    x2 = layers.MaxPooling2D((2, 2))(x1)  # 24 (64)

    x3 = inception_cluster(x=x2, num=64)  # 24 (64)

    x4 = layers.Concatenate()([x2, x3])  # 24 (64 + 64)

    x5 = layers.UpSampling2D(size=(2, 2))(x4)  # 48 (128)

    x6 = layers.Conv2D(64, (1, 1), activation="relu")(x5)  # 48 (64)

    x7 = inception_cluster(x=x6, num=64)  # 48 (64)

    x8 = layers.Concatenate()([x, x5, x7])  # 48 (64 + 128 + 64)

    x9 = layers.Conv2D(128, (1, 1), activation="relu")(x8)  # 48 (128)

    x10 = layers.UpSampling2D(size=(2, 2))(x9)  # 96 (128)

    x11 = basic_cluster(x=x10, num=64)  # 96 (64)

    x12 = tf.pad(x11, paddings, "SYMMETRIC")  # 98

    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid")(x12)  # 96

    # model done
    model = keras.Model(Input_img, decoded)
    return model
