import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19, preprocess_input


def set_image_size(size):
    if size < 32:
        raise "Size cannot be lower than 32 pixels"
    global image_size
    image_size = size
    vgg_style.model = None
    vgg_content.model = None


def vgg_style(X, Y):
    if not vgg_style.model:
        vgg_model = VGG19(include_top=False, input_shape=(image_size, image_size, 3))
        vgg_model.trainable = False

        layerNames = [[1, 2], [2, 2], [3, 4], [4, 4], [5, 4]]

        Xx = vgg_model.input
        Yy = [
            vgg_model.get_layer(name=f"block{i[0]}_conv{i[1]}").output
            for i in layerNames
        ]
        vgg_style.model = keras.Model(Xx, Yy)

    Xt = preprocess_input(X * 255)
    Yt = preprocess_input(Y * 255)

    vggX = vgg_style.model(Xt)
    vggY = vgg_style.model(Yt)

    ret = 0
    for x, y in zip(vggX, vggY):
        ret += tf.reduce_mean(tf.square(x - y))

    return ret / len(vggX)


def vgg_content(X, Y):
    if not vgg_content.model:
        vgg_model = VGG19(include_top=False, input_shape=(image_size, image_size, 3))
        vgg_model.trainable = False

        Xx = vgg_model.input
        Yy = vgg_model.get_layer(name=f"block5_conv4").output
        vgg_content.model = keras.models.Model(Xx, Yy)

    Xt = preprocess_input(X * 255)
    Yt = preprocess_input(Y * 255)

    vggX = vgg_content.model(Xt)
    vggY = vgg_content.model(Yt)

    return tf.reduce_mean(tf.square(vggY - vggX))
