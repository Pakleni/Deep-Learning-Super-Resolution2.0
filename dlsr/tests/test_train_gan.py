from .. import *


def test_train_gan():
    discriminator = models.discriminator()
    generator = models.basic()

    training_data = helpers.get_training_data(num=8)

    generator, gen_his, discriminator, dis_his = train_gan(
        discriminator=discriminator,
        generator=generator,
        training_data=training_data,
        epochs=3,
        discriminator_epochs=5,
        discriminator_patience=3,
        discriminator_batch_size=8,
        discriminator_n=1e-6,
        generator_epochs=3,
        generator_loss_fn=losses.get_srgan_loss(discriminator)["srgan_vgg_loss"],
        generator_patience=1,
        generator_batch_size=8,
        generator_n=1e-6,
        pre_train_epochs=2,
        pre_train=True,
        pre_train_loss_fn=losses.vgg_style_loss,
    )

    generator.save("./saved-models/tests/gan/generator.h5")
    discriminator.save("./saved-models/tests/gan/discriminator.h5")

    helpers.plot_history(gen_his)
    helpers.plot_history(dis_his)
