import tensorflow as tf
from keras import backend as K

from .. import helpers


def ssim(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, helpers.normalize(255)))


def psnr_mse(y_true, y_pred):
    max_pixel = helpers.normalize(255)
    return (
        -(
            10.0
            * K.log(
                (max_pixel**2) / (K.mean(K.square(y_pred - y_true + 1e-8), axis=-1))
            )
        )
        / 2.303
    )


def psnr_mae(y_true, y_pred):
    max_pixel = helpers.normalize(255)
    return (
        -(
            10.0
            * K.log((max_pixel**2) / (K.mean(K.abs(y_pred - y_true + 1e-8), axis=-1)))
        )
        / 2.303
    )
