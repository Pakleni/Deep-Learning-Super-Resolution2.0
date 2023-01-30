import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19, preprocess_input


class StyleLoss(keras.losses.Loss):
    def __init__(self, image_size, name="style_loss"):
        super().__init__(name=name)

        vgg_model = VGG19(include_top=False, input_shape=(image_size, image_size, 3))
        vgg_model.trainable = False

        layerNames = [[1, 2], [2, 2], [3, 4], [4, 4], [5, 4]]

        Xx = vgg_model.input
        Yy = [
            vgg_model.get_layer(name=f"block{i[0]}_conv{i[1]}").output
            for i in layerNames
        ]
        self.model = keras.Model(Xx, Yy)

    def process(self, x):
        x = x * 127.5 + 127.5
        x = preprocess_input(x)
        x = self.model(x, training=False)
        return x

    def call(self, y_true, y_pred):
        vggX = self.process(y_true)
        vggY = self.process(y_pred)

        layers = len(self.model.output_shape)
        total = 0
        for x, y in zip(vggX, vggY):
            total += tf.reduce_mean(tf.square(x - y), axis=[1, 2, 3])

        return total / layers


class ContentLoss(keras.losses.Loss):
    def __init__(self, image_size, name="content_loss"):
        super().__init__(name=name)

        vgg_model = VGG19(include_top=False, input_shape=(image_size, image_size, 3))
        vgg_model.trainable = False

        Xx = vgg_model.input
        Yy = vgg_model.get_layer(name=f"block5_conv4").output
        self.model = keras.models.Model(Xx, Yy)

    def process(self, x):
        x = x * 127.5 + 127.5
        x = preprocess_input(x)
        x = self.model(x, training=False)
        return x

    def call(self, y_true, y_pred):
        vggX = self.process(y_true)
        vggY = self.process(y_pred)

        return tf.reduce_sum(tf.square(vggY - vggX), axis=[1, 2]) / (
            self.model.output_shape[1] * self.model.output_shape[2]
        )
