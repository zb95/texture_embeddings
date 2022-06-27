import tensorflow as tf
from tensorflow import keras


class RandomHue(keras.layers.Layer):
    def __init__(self, max_delta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.rng = tf.random.Generator.from_seed(123, alg='philox')

    def call(self, inputs):
        seed = self.rng.make_seeds(2)
        return tf.image.stateless_random_hue(inputs, self.max_delta, seed[0])


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class AdjustContrast(keras.layers.Layer):
    def __init__(self, factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs):
        return tf.image.adjust_contrast(inputs, self.factor)