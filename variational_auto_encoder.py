##################
#Keras variational autoencoder
#based on sources under with only minor changes
#https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb#scrollTo=iYn4MdZnKCey
#https://www.tensorflow.org/tutorials/generative/cvae
##################

# import the necessary packages
from cbof import BoF_Pooling
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

#from IPython import display

from tensorflow.keras.layers import Lambda

class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(64, 64, 1)),

          tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D(2, 2),

          tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D(2, 2),

          tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D(2, 2),

          tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D(2, 2),


          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=4*4*32, activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(4, 4, 32)),
          tf.keras.layers.UpSampling2D(size=(2, 2)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding="SAME",activation='relu'),

          tf.keras.layers.UpSampling2D(size=(2, 2)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding="SAME",activation='relu'),

          tf.keras.layers.UpSampling2D(size=(2, 2)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding="SAME",activation='relu'),

          tf.keras.layers.UpSampling2D(size=(2, 2)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="SAME",activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits

optimizer = tf.keras.optimizers.Adam(1e-3)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 100
latent_dim = 64
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

generate_and_save_images(model, 0, random_vector_for_generation)

def vae(x_train,x_test):
    x_train  = x_train.reshape(x_train.shape[0], 64, 64, 1).astype('float32')
    x_test  = x_test.reshape(x_test.shape[0], 64, 64, 1).astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(30000).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(10000).batch(32)
    for epoch in range(1, epochs + 1):
      start_time = time.time()
      for train_x in train_dataset:
        compute_apply_gradients(model, train_x, optimizer)
      end_time = time.time()

      if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
          loss(compute_loss(model, test_x))
        elbo = -loss.result()
        #display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
    generate_and_save_images(
            model, epoch, random_vector_for_generation)
    return model
