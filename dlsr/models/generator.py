import tensorflow as tf

from tensorflow import keras

from keras import layers

# Because of pixel shuffle issue and because ESRGAN uses UpSampling we decided to
# chop off the last few layers of SRGAN and replace them with how ESRGAN does it
# We removed Batch Normalization because of a claim in the SRGAN papaer
# We removed one group of upscaling layers so that we could only achieve x2 superresolution


def res():
    def run(x):
        skip = x
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
        # x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Add()([x, skip])
        return x

    return run


# B = 16 is used in the SRGAN folder
def generator(B=16):

    Input_img = keras.Input(shape=(None, None, 3))
    x = Input_img

    x = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)

    skip = x

    for i in range(B):
        x = res()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])

    # x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    # x = tf.nn.depth_to_space(x, 2, data_format="NHWC", name=None)
    # x = layers.PReLU(shared_axes=[1, 2])(x)

    # x = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    # x = tf.nn.depth_to_space(x, 2, data_format="NHWC", name=None)
    # x = layers.PReLU(shared_axes=[1, 2])(x)

    # decoded = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(x)

    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    decoded = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same")(x)

    return keras.Model(Input_img, decoded)
