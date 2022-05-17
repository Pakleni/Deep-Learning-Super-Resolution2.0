from dlsr import *

helpers.config(True)

model = models.basic()

training_data = helpers.get_training_data(num=3600)

model, history = train(
    model,
    training_data,
    batch_size=15,
    epochs=20,
    loss_fn=losses.vgg_style_loss,
    patience=2,
    n=1e-6,
)

model.save("./saved-models/vgg_style.h5")
helpers.plot_history(history)
