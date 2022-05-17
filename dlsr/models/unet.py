from tensorflow import keras
from keras import layers

from . import custom_layers


def unet(frames=128):

    Input_img = keras.Input(shape=(None, None, 3))  # 48

    x1 = custom_layers.down(x=Input_img, frames=frames // 16)

    x2 = layers.MaxPooling2D((2, 2))(x1)  # 24
    x2 = custom_layers.down(x=x2, frames=frames // 8)

    x3 = layers.MaxPooling2D((2, 2))(x2)  # 12
    x3 = custom_layers.down(x=x3, frames=frames // 4)

    x4 = layers.MaxPooling2D((2, 2))(x3)  # 6
    x4 = custom_layers.down(x=x4, frames=frames // 2)

    x5 = custom_layers.up(frames=frames // 2, x=x4)  # 12
    x5 = layers.Concatenate()([x3, x5])
    x5 = custom_layers.basic_cluster(x=x5, frames=frames)

    x6 = custom_layers.up(frames=frames // 4, x=x5)  # 24
    x6 = layers.Concatenate()([x2, x6])
    x6 = custom_layers.basic_cluster(x=x6, frames=frames // 2)

    x7 = custom_layers.up(frames=frames // 8, x=x6)  # 48
    x7 = layers.Concatenate()([x1, x7])  # 48
    x7 = custom_layers.basic_cluster(x=x7, frames=frames // 4)

    x8 = custom_layers.up(frames=frames // 16, x=x7)  # 96
    x8 = custom_layers.basic_cluster(x=x8, frames=frames // 8)

    decoded = custom_layers.padded_conv(x8, 3, (3, 3), activation="sigmoid")  # 96

    return keras.Model(Input_img, decoded)
