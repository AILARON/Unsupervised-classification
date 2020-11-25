##################
#Tensorflow-keras based implementation of Generative Adverserial Network (GAN)

#This file is based on the tutorial presented at the following link:
#https://www.tensorflow.org/tutorials/generative/dcgan

#Note 1. that the code is adapted to work with the kaggle images.
#Performance might therefore differ.

##################

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from tensorflow.keras import backend

import cv2
from preprocessing import Preprocessing, PreprocessingFromDataframe

from load_dataset import importKaggle
from clustering_algorithms import TSNEAlgo, PCAAlgo, KMeansCluster, SpectralCluster

from neural_network_utils import saveWeights, loadWeights, compute_features

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# calculate wasserstein loss
def wasserstein_loss(y_pred,y_true):
    return backend.mean(y_true * y_pred)


#CSV load files
KAGGLE_TRAIN = 'csvloadfiles/kaggle_original.csv'
KAGGLE_TEST = 'csvloadfiles/kaggle_five_classes.csv'
KAGGLE_MISSING = 'csvloadfiles/kaggle_missing.csv'
KAGGLE_MISSING_TEST = 'csvloadfiles/kaggle_missing_five.csv'



class GAN():

    NUM_CLUSTER = 121

    input_shape = (64,64,3)

    EPOCHS = 1
    noise_dim = 100
    batch_size = 32
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".


    def __init__(self):
        return

    def make_generator_model(self):
        model = tf.keras.Sequential(name='generator')
        model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 256)))
        assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 3)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential( name='discriminator')
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[64, 64, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.GlobalAveragePooling2D())
        #model.add(layers.Flatten())

        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1))

        return model




    def train(self,dataset, epochs,  generator,generator_optimizer, discriminator, discriminator_optimizer):
        @tf.function
        def train_step(images, generator,generator_optimizer, discriminator,discriminator_optimizer):
            noise = tf.random.normal([self.batch_size, self.noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)


            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        #model_wrapper = wrapper()
        for epoch in range(epochs):

            start = time.time()

            for image_batch, labels in dataset:
                train_step(image_batch, generator,generator_optimizer, discriminator,discriminator_optimizer)


            if epoch % 10 == 0:

                self.generate_and_save_images(generator,
                                     epoch + 1,
                                     self.seed)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    def generate_and_save_images(self,model, epoch, test_input):

      predictions = model(test_input, training=False)

      fig = plt.figure(figsize=(4,4))

      for i in range(predictions.shape[0]):
          plt.subplot(4, 4, i+1)
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
          plt.axis('off')

      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


    def train_gan(self):

        ##### Load training data #####
        train_data, _ = importKaggle(train=True)
        labels = np.zeros(train_data.shape[0])
        ##### Preprocessing #####

        preprocess_training = Preprocessing(train_data,labels,dataset='Kaggle',
         num_classes = self.NUM_CLUSTER,input_shape = self.input_shape)
        train_dataset = preprocess_training.returnTrainDataset()

        #make model
        generator = self.make_generator_model()
        discriminator = self.make_discriminator_model()
        print(discriminator.summary())
        print(generator.summary())

        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


        self.train(train_dataset, 50, generator,generator_optimizer, discriminator,discriminator_optimizer)
        saveWeights(discriminator,"discriminator")
        saveWeights(generator,"generator")


    def test_gan(self):
        discriminator = self.make_discriminator_model()
        loadWeights(discriminator,"discriminator")

        discriminator_pred = tf.keras.Model(inputs=discriminator.input,
                                          outputs=discriminator.layers[-2].output)

        val_data, val_label = importKaggle(train=False)
        preprocess = Preprocessing(val_data,val_label,dataset='Kaggle',
         num_classes = self.NUM_CLUSTER,input_shape = self.input_shape)
        validate_dataset = preprocess.returnDataset()
        images = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (val_data)])
        features = compute_features(validate_dataset, discriminator_pred,val_data.shape[0])


        #Predict using K-means
        print("KMEANS CLUSTERING")
        kmean = KMeansCluster(n_clusters = 5)
        kmean.fit(features)
        k_means_labels = kmean.sortLabels(val_label,kmean.predict(features))
        accuracy, precision, recall, f1score = kmean.performance(val_label,k_means_labels)

        print("Spectral CLUSTERING")
        spectral = SpectralCluster(n_clusters = 5)
        spectral.fit(features)
        spectral_labels = spectral.sortLabels(val_label,spectral.predict(features))
        accuracy, precision, recall, f1score = spectral.performance(val_label,spectral_labels)

        #Visualize using TSNE
        TSNE = TSNEAlgo()
        TSNE.tsne_fit(features,perplexity = 35)

        TSNE.tsne_plot(images,val_label,"baseline","baseline")
        TSNE.tsne_plot(images,k_means_labels,"kmeans","baseline")
        TSNE.tsne_plot(images,spectral_labels,"spectral","baseline")

        anim_file = 'dcgan.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
          filenames = glob.glob('image*.png')
          filenames = sorted(filenames)
          for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
          image = imageio.imread(filename)
          writer.append_data(image)

        def testSimilarImage(self):

            discriminator = loadWeights(discriminator)
            generator = loadWeights(generator)
            doesImageExist(train_images,generator)

def doesImageExist(data,model):

    data_np = data.reshape(-1,64*64)
    print(data_np.shape)


    seed = tf.random.normal([16, 100])
    predictions = model(seed, training=False)
    print(predictions.shape)

    pred_np = predictions.numpy()
    pred_np = pred_np.reshape(-1,64*64)
    print(pred_np.shape)



    from sklearn.neighbors import DistanceMetric
    dist = DistanceMetric.get_metric('manhattan')
    array = dist.pairwise(pred_np, Y=data_np)
    print(array.shape)
    index = np.zeros(16)
    for i in range(array.shape[0]):
        print(np.amin(array[i]))
        print(np.where(array[i] == np.amin(array[i])))
        index[i] = np.where(array[i] == np.amin(array[i]))[0]
    index = index.astype('int')
    print(index)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('generated.png')

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(data[index[i], :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('similar.png')



    sys.exit()
