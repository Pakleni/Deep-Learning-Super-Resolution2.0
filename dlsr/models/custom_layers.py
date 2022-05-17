import tensorflow as tf
from keras import layers


def padded_conv(x, frames, size, activation):
    if not padded_conv.paddings:
        padded_conv.paddings = {
            (3, 3): tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]),
            (5, 5): tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]),
            (7, 7): tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]]),
        }

    if size != (1, 1):
        x = tf.pad(x, padded_conv.paddings[size], "SYMMETRIC")

    x = layers.Conv2D(frames, size, activation=activation)(x)
    return x


padded_conv.paddings = None


def basic_cluster(x, frames):
    x = padded_conv(x, frames, (3, 3), "relu")
    x = padded_conv(x, frames, (3, 3), "relu")
    x = padded_conv(x, frames, (3, 3), "relu")
    x = padded_conv(x, frames // 2, (1, 1), "relu")
    return x


def down(x, frames):
    x = padded_conv(x, frames, (3, 3), "relu")
    x = padded_conv(x, frames, (3, 3), "relu")
    x = padded_conv(x, frames, (3, 3), "relu")
    x = padded_conv(x, frames * 2, (1, 1), "relu")
    return x


def up(x, frames):
    # x = layers.Conv2DTranspose(frames, (2,2),strides=(2,2), activation = 'relu')(x)
    x = layers.Conv2D(frames, (1, 1), activation="relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    return x


def inception_cluster(x, frames):
    x1 = x
    x1_1 = padded_conv(x, frames / 4, (1, 1), "relu")
    x1_2 = padded_conv(x, frames / 4, (1, 1), "relu")
    x1_3 = padded_conv(x, frames / 4, (1, 1), "relu")

    x2_1 = padded_conv(x1, frames / 4, (1, 1), activation="relu")
    x2_2 = padded_conv(x1_1, frames / 4, (3, 3), activation="relu")
    x2_3 = padded_conv(x1_2, frames / 4, (5, 5), activation="relu")
    x2_4 = padded_conv(x1_3, frames / 4, (7, 7), activation="relu")

    x2 = layers.Concatenate()([x2_1, x2_2, x2_3, x2_4])

    return x2


def res(x, frames):

    x_temp = x

    x = padded_conv(x, frames, (3, 3), activation="relu")

    x = padded_conv(x, frames, (3, 3), activation="relu")

    x = layers.Add()([x, x_temp])

    return x
