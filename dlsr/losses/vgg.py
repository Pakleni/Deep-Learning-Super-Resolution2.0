import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19, preprocess_input

image_size = 96


def set_image_size(size):
    if size < 32:
        raise "Size cannot be lower than 32 pixels"
    global image_size
    image_size = size
    vgg_loss.vgg_model = None
    vgg_style_loss.layers = None


def vgg_loss(X, Y):
    if not vgg_loss.vgg_model:
        vgg_loss.vgg_model = VGG19(
            include_top=False, input_shape=(image_size, image_size, 3)
        )

    Xt = preprocess_input(X * 255)
    Yt = preprocess_input(Y * 255)

    vggX = vgg_loss.vgg_model(Xt)
    vggY = vgg_loss.vgg_model(Yt)

    return tf.reduce_mean(tf.square(vggY - vggX))


vgg_loss.vgg_model = None


def vgg_style_loss(X, Y):
    if not vgg_style_loss.layers:
        vgg_model = VGG19(include_top=False, input_shape=(image_size, image_size, 3))
        layerNames = [[1, 2], [2, 2], [3, 4], [4, 4], [5, 4]]
        Xx = vgg_model.input
        vgg_style_loss.layers = []
        for i in layerNames:
            Yy = vgg_model.get_layer(name=f"block{i[0]}_conv{i[1]}").output
            vgg_style_loss.layers.append(keras.models.Model(Xx, Yy))

    Xt = preprocess_input(X * 255)
    Yt = preprocess_input(Y * 255)

    ret = 0
    for curr in vgg_style_loss.layers:
        vggX = curr(Xt)
        vggY = curr(Yt)

        ret += tf.reduce_mean(tf.square(vggY - vggX)) / (
            curr.output_shape[1] * curr.output_shape[2] * curr.output_shape[3]
        )

    return ret


vgg_style_loss.layers = None
