import tensorflow as tf

from .vgg import vgg_content_loss


def get_srgan_loss(srgan):
    def gan_loss(y_true, y_pred):
        # this is not that applicable to SRGAN
        return srgan(y_pred)

    def srgan_loss(y_true, y_pred):
        return 1e-3 * (
            -tf.math.log(gan_loss(y_true, y_pred) + 1e-8)
        ) + 1 / 12.75 * vgg_content_loss(y_true, y_pred)

    return {"gan_loss": gan_loss, "srgan_loss": srgan_loss}
