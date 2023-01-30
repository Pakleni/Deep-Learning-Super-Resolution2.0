import tensorflow as tf

from tensorflow import keras


class SRGAN(keras.models.Model):
    def __init__(self, discriminator, generator, *args):
        super(SRGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.g_accuracy = keras.metrics.CategoricalAccuracy(name="g_accuracy")
        self.d_accuracy = keras.metrics.BinaryAccuracy(name="d_accuracy")

    def compile(self, d_optimizer, g_optimizer, d_loss, g_loss):
        super(SRGAN, self).compile()
        # desciminator setup
        self.d_optimizer = d_optimizer
        self.d_loss = d_loss
        # generator setup
        self.g_optimizer = g_optimizer
        self.g_loss = g_loss

    def call(self, inputs, training=None, mask=None):
        return self.generator(inputs, training=training)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.g_accuracy, self.d_accuracy]

    def test_step(self, data):
        # Unpack the data
        lr, hr = data

        batch_size = tf.shape(lr)[0]

        # Generate fake images
        generated_images = self.generator(lr, training=False)
        g_loss = self.g_loss(hr, generated_images)
        self.g_accuracy.update_state(hr, generated_images)

        # Combine them with real images
        combined_images = tf.concat([generated_images, hr], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )
        predictions = self.discriminator(combined_images, training=False)
        d_loss = self.d_loss(labels, predictions)
        self.d_accuracy.update_state(labels, predictions)

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_accuracy": self.g_accuracy.result(),
            "d_accuracy": self.d_accuracy.result(),
        }

    def train_step(self, data):
        lr, hr = data

        batch_size = tf.shape(lr)[0]

        # Generate fake images
        generated_images = self.generator(lr, training=False)

        # Combine them with real images
        combined_images = tf.concat([generated_images, hr], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.d_loss(labels, predictions)
        self.d_accuracy.update_state(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        with tf.GradientTape() as tape:
            predictions = self.generator(lr)
            g_loss = self.g_loss(hr, predictions)
        self.g_accuracy.update_state(hr, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_accuracy": self.g_accuracy.result(),
            "d_accuracy": self.d_accuracy.result(),
        }
