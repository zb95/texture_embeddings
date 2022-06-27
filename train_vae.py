import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation,\
    RandomZoom, Resizing, RandomCrop
import helper as hlp
import layers as ls
import models as my_models
import time


# fix for CUDNN_STATUS_INTERNAL_ERROR
# comment out if TensorFlow runs on CPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# region model definition ##########################################################
# load your dataset of textures here
dataset = hlp.unpickle('textures.pkl')

train_images = dataset['data_tr']
test_images = dataset['data_tst']
# endregion ##########################################################

# region model definition ##########################################################
l2_scale = 0.00000
beta = 0.000001  # scales the KL-divergence loss
gamma = 0.0003  # scales the similarity loss, similar to the technique used in SimCLR
epochs = 1000
batch_size = 64
intermediate_dim = 1024
latent_dim = 512

input_shape = train_images[0].shape
random_crop_shape = (50, 50, 3)
encoder_input_shape = (32, 32, 3)

# flow during training: augmentor -> resizer_tr -> enhancer (optional) -> encoder -> decoder
# flow during test: resizer_tst -> enhancer (optional) -> encoder -> decoder

augmentor_inputs = tf.keras.Input(shape=input_shape)
x = RandomFlip()(augmentor_inputs)
x = RandomZoom((-0.3, 0), (-0.3, 0))(x)
x = RandomRotation(factor=0.2)(x)
x = RandomCrop(random_crop_shape[0], random_crop_shape[1])(x)
x = ls.RandomHue(max_delta=0.1)(x)
augmentor_outputs = x
augmentor = tf.keras.Model(augmentor_inputs, augmentor_outputs, name="augmentor")
augmentor.summary()

# during training, resizes augmented images to be fed to the encoder/fixed augmentor
resizer_tr_inputs = tf.keras.Input(random_crop_shape)
x = Resizing(encoder_input_shape[0], encoder_input_shape[1], interpolation='lanczos3')(resizer_tr_inputs)
resizer_tr_outputs = x
resizer_tr = tf.keras.Model(resizer_tr_inputs, resizer_tr_outputs, name="resizer_tr")

# during inference, resizes input images to be fed to the encoder/enhancer
resizer_tst_inputs = tf.keras.Input(shape=input_shape)
x = Resizing(encoder_input_shape[0], encoder_input_shape[1], interpolation='lanczos3')(resizer_tst_inputs)
resizer_tst_outputs = x
resizer_tst = tf.keras.Model(resizer_tst_inputs, resizer_tst_outputs, name="resizer_tst")

# the enhancer sits right before the encoder and serves to make texture details more distinct
enhancer_inputs = tf.keras.Input(encoder_input_shape)
x = ls.AdjustContrast(factor=1.5)(enhancer_inputs)
enhancer_outputs = x
enhancer = tf.keras.Model(enhancer_inputs, enhancer_outputs, name="enhancer")
enhancer.summary()

# encoder
encoder_inputs = tf.keras.Input(shape=encoder_input_shape)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", kernel_regularizer=regularizers.L2(l2_scale))(
    encoder_inputs)
x = layers.Conv2D(128, 4, activation="relu", strides=2, padding="same", kernel_regularizer=regularizers.L2(l2_scale))(x)
x = layers.Conv2D(256, 4, activation="relu", strides=2, padding="same", kernel_regularizer=regularizers.L2(l2_scale))(x)
x = layers.Flatten()(x)
x = layers.Dense(intermediate_dim, activation="relu", kernel_regularizer=regularizers.L2(l2_scale))(x)
z_mean = layers.Dense(latent_dim, name="z_mean", kernel_regularizer=regularizers.L2(l2_scale))(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var", kernel_regularizer=regularizers.L2(l2_scale))(x)
z = ls.Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 64, activation="relu", kernel_regularizer=regularizers.L2(l2_scale))(latent_inputs)
x = layers.Reshape((8, 8, 64))(x)
x = layers.Conv2DTranspose(256, 4, activation="relu", strides=2, padding="same",
                           kernel_regularizer=regularizers.L2(l2_scale))(x)
x = layers.Conv2DTranspose(128, 4, activation="relu", strides=2, padding="same",
                           kernel_regularizer=regularizers.L2(l2_scale))(x)
decoder_outputs = layers.Conv2DTranspose(3, 4, activation="sigmoid", padding="same",
                                         kernel_regularizer=regularizers.L2(l2_scale))(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

model = my_models.TextureVAE(encoder, decoder, augmentor=augmentor, enhancer=enhancer,
                             resizer_tr=resizer_tr, resizer_tst=resizer_tst, beta=beta, gamma=gamma)
model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)
# endregion ##########################################################

# region training ##########################################################
start_time = time.time()
model.fit(train_images, validation_data=[test_images], epochs=epochs, batch_size=batch_size)

print('time taken', time.time() - start_time)
# endregion ##########################################################

# region computing embeddings & displaying closest matches ##########################################################
# For the computation of embeddings, we are feeding the model appropriately sized patches of images,
# so there is no need for resizing
model.resizer_tst = None
# compute the average of the embeddings of 9 evenly spaced patches
e = hlp.avg_embeddings_patches(test_images, 9, encoder_input_shape[:2], model)
# endregion ##########################################################
