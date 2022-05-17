import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def resnet():
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    def res(x, num):

        x_temp = x

        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num, (3, 3), activation="relu")(x)

        x = tf.pad(x, paddings, "SYMMETRIC")
        x = layers.Conv2D(num, (3, 3), activation="relu")(x)

        x = layers.Add()([x, x_temp])

        return x

    # input layer
    Input_img = keras.Input(shape=(48, 48, 3))  # 48

    x = tf.pad(Input_img, paddings, "SYMMETRIC")
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)

    x = res(x=x, num=64)
    x = res(x=x, num=64)
    x = res(x=x, num=64)
    x = res(x=x, num=64)
    x = res(x=x, num=64)
    x = res(x=x, num=64)

    x = layers.UpSampling2D(size=(2, 2))(x)  # 96

    x = layers.Conv2D(32, (1, 1), activation="relu")(x)

    x = res(x=x, num=32)
    x = res(x=x, num=32)
    x = res(x=x, num=32)
    x = res(x=x, num=32)
    x = res(x=x, num=32)
    x = res(x=x, num=32)

    x = tf.pad(x, paddings, "SYMMETRIC")  # 98
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid")(x)  # 96

    # model done
    model = keras.Model(Input_img, decoded)
    return model
