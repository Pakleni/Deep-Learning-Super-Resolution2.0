import tensorflow as tf
from keras import backend as K

from .. import helpers


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, helpers.norm(255)))


def psnr_loss(y_true, y_pred):
    max_pixel = helpers.norm(255)
    return (
        -(
            10.0
            * K.log(
                (max_pixel**2) / (K.mean(K.square(y_pred - y_true + 1e-8), axis=-1))
            )
        )
        / 2.303
    )


def psnr_abs_loss(y_true, y_pred):
    max_pixel = helpers.norm(255)
    return (
        -(
            10.0
            * K.log((max_pixel**2) / (K.mean(K.abs(y_pred - y_true + 1e-8), axis=-1)))
        )
        / 2.303
    )
