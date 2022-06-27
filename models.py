import tensorflow as tf


# TODO description of the flow
class TextureVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1, gamma=0., augmentor=None, enhancer=None,
                 resizer_tr=None, resizer_tst=None, **kwargs):
        super(TextureVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.gamma = gamma
        self.augmentor = augmentor
        self.enhancer = enhancer
        self.resizer_tr = resizer_tr
        self.resizer_tst = resizer_tst

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        if self.augmentor is not None and self.gamma > 0.:
            self.similarity_loss_tracker = tf.keras.metrics.Mean(name="sim_loss")

    @property
    def metrics(self):
        if self.augmentor is not None and self.gamma > 0.:
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.similarity_loss_tracker
            ]
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        xs = data
        xs_compare = data
        with tf.GradientTape() as tape:
            if self.augmentor is not None:
                xs = self.augmentor(xs)
                xs_compare = self.augmentor(xs_compare)
            if self.resizer_tr is not None:
                xs = self.resizer_tr(xs)
                xs_compare = self.resizer_tr(xs_compare)
            if self.enhancer is not None:
                xs = self.enhancer(xs)
                xs_compare = self.enhancer(xs_compare)
            z_mean, z_log_var, z = self.encoder(xs)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.square(xs - reconstruction))
            _, _, z_compare = self.encoder(xs_compare)
            similarity_loss = 0.
            if self.augmentor is not None and self.gamma > 0.:
                similarity_loss = tf.reduce_mean(tf.norm(z - z_compare, 2, axis=1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta*kl_loss + self.gamma*similarity_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if self.augmentor is not None and self.gamma > 0.:
            self.similarity_loss_tracker.update_state(similarity_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "sim_loss": self.similarity_loss_tracker.result()
            }
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        xs = data
        if self.resizer_tst is not None:
            xs = self.resizer_tst(xs)
        if self.enhancer is not None:
            xs = self.enhancer(xs)
        z_mean, z_log_var, z = self.encoder(xs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.square(xs - reconstruction))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "val. loss": self.total_loss_tracker.result(),
            "val. rec_loss": self.reconstruction_loss_tracker.result(),
            "val. kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, x, training=None, mask=None):
        x_ = x
        if self.resizer_tst is not None:
            x_ = self.resizer_tst(x_)
        z = self.encoder(x_)[2]
        reconstruction = self.decoder(z)
        return reconstruction

    def get_embeddings(self, x):
        x_ = x
        if self.enhancer is not None:
            x_ = self.enhancer(x_)
        return self.encoder(x_)[0]

    def get_config(self):
        return {"encoder": self.encoder, "decoder": self.decoder,
                "beta": self.beta, "augmentor": self.augmentor, "resizer": self.resizer_tst}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
