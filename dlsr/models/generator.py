import tensorflow as tf

from tensorflow import keras

from keras import layers


def res():
    def run(x):
        skip = x
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, skip])
        return x

    return run


# B = 16 is used in the SRGAN folder
def generator(B=16, scale=2) -> keras.models.Model:

    if scale != 2 and scale != 4:
        raise ValueError("Scale must be 2 or 4")

    Input_img = keras.Input(shape=(None, None, 3))
    x = Input_img

    x = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    skip = x

    for i in range(B):
        x = res()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])

    if scale == 4:
        x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
        x = tf.nn.depth_to_space(x, 2, data_format="NHWC", name=None)
        x = layers.PReLU(shared_axes=[1, 2])(x)

    x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = tf.nn.depth_to_space(x, 2, data_format="NHWC", name=None)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    decoded = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(x)

    return keras.Model(Input_img, decoded)
