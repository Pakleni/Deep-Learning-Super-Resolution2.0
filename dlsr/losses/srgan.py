import tensorflow as tf

from .vgg import vgg_content


def get_srgan_loss(discriminator):
    def adverserial(y_true, y_pred):
        return discriminator(y_pred)

    def perceptual(y_true, y_pred):
        return 1e-3 * (
            -tf.math.log(adverserial(y_true, y_pred) + 1e-8)
        ) + 1 / 12.75 * vgg_content(y_true, y_pred)

    return {"adverserial": adverserial, "perceptual": perceptual}
