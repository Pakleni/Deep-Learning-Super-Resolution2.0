import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from .. import helpers


class SSIM(keras.losses.Loss):
    def __init__(self, name="ssim"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return 1 - tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, helpers.normalize_hr(255))
        )


class PSNR_MSE(keras.losses.Loss):
    def __init__(self, name="psnr_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        max_pixel = helpers.normalize_hr(255)
        return (
            -(
                10.0
                * K.log(
                    (max_pixel**2)
                    / (K.mean(K.square(y_pred - y_true + 1e-8), axis=-1))
                )
            )
            / 2.303
        )


class PSNR_MAE(keras.losses.Loss):
    def __init__(self, name="psnr_mae"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        max_pixel = helpers.normalize_hr(255)
        return (
            -(
                10.0
                * K.log(
                    (max_pixel**2) / (K.mean(K.abs(y_pred - y_true + 1e-8), axis=-1))
                )
            )
            / 2.303
        )
