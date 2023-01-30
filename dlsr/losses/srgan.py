import tensorflow as tf
from tensorflow import keras
from .vgg import ContentLoss


class PerceptualLoss(keras.losses.Loss):
    adverserial_weight = 1e-3
    content_weight = 1 / 12.75 * 1 / 12.75

    def adverserial_loss(self, y_true, y_pred):
        return -tf.math.log(
            self.discriminator(y_pred, training=False) + keras.backend.epsilon()
        )

    def __init__(self, discriminator, image_size, name="perceptual_loss"):
        super().__init__(name=name)
        self.discriminator = discriminator
        self.content_loss = ContentLoss(image_size)

    def call(self, y_true, y_pred):
        return self.adverserial_weight * self.adverserial_loss(
            y_true, y_pred
        ) + self.content_weight * self.content_loss(y_true, y_pred)
