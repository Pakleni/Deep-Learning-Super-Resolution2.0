from .. import *


def test_train():
    model = models.basic()

    training_data = helpers.get_training_data(num=15)

    history = train(
        model,
        training_data,
        batch_size=15,
        epochs=2,
        loss_fn=losses.vgg_style_loss,
        patience=1,
        n=1e-5,
    )

    model.save("./saved-models/tests/model.h5")
    helpers.plot_history(history)


def test_train_from_file():
    import tensorflow as tf

    model = tf.keras.models.load_model(
        "./saved-models/tests/model.h5", custom_objects=losses.get_custom_objects()
    )

    training_data = helpers.get_training_data(num=15)

    history = train(
        model,
        training_data,
        batch_size=15,
        epochs=2,
        patience=1,
        n=1e-5,
    )

    model.save("./saved-models/tests/model.h5")
    helpers.plot_history(history)
