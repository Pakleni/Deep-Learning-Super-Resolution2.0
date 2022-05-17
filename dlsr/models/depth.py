import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def depth():
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    def down(x, num):
        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num / 2, (3, 3), activation="relu")(x)

        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num / 2, (3, 3), activation="relu")(x)

        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num / 2, (3, 3), activation="relu")(x)

        x = layers.Conv2D(num, (1, 1), activation="relu")(x)

        return x

    # input layer
    Input_img = keras.Input(shape=(48, 48, 3))  # 48

    x = down(x=Input_img, num=512)
    x = down(x=x, num=1024)

    x = tf.pad(x, paddings, "SYMMETRIC")
    x = layers.Conv2D(3 * (2**2), (3, 3), activation="sigmoid")(x)

    decoded = tf.nn.depth_to_space(x, 2, data_format="NHWC", name=None)

    # model done
    model = keras.Model(Input_img, decoded)
    return model
