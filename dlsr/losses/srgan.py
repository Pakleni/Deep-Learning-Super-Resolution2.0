import tensorflow as tf

from keras import backend as K
from .basic import ssim_loss
from .vgg import vgg_style_loss


def get_srgan_loss(srgan):
    def srgan_loss(y_true, y_pred):
        # this is not that applicable to SRGAN
        return srgan(y_pred)

    def srgan_ssim_loss(y_true, y_pred):
        # this is a test function for which I have found no real use
        return -K.log(1 - srgan_loss(y_true, y_pred) + 1e-8) * 1e-2 + ssim_loss(
            y_true, y_pred
        )

    def srgan_vgg_loss(y_true, y_pred):
        return -K.log(1 - srgan_loss(y_true, y_pred) + 1e-8) * 1e-3 + vgg_style_loss(
            y_true, y_pred
        )

    return {
        "srgan_loss": srgan_loss,
        "srgan_ssim_loss": srgan_ssim_loss,
        "srgan_vgg_loss": srgan_vgg_loss,
    }
