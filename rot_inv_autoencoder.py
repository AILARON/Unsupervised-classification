#####################
# Class implementation of rotation invariant autoencoder
# The code is using the Cohen et al. rotation invariant layer from https://github.com/tscohen/gconv_experiments
# and the keras implementation provided by Veeling et al. https://github.com/basveeling/keras-gcnn
#
# The autoencoder is inspired by the rotation invariant autoencoder from the Kuzminykh et al. paper 'Extracting Invariant Features From Images
# Using An Equivariant Autoencoder'
#
####################


import tensorflow as tf
import os
import keras
import cv2
import numpy as np
from numpy.random import randint
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras_gcnn.layers import GConv2D, GBatchNorm
from keras_gcnn.layers.pooling import GroupPool
from utils import visualize_class_activation_map
from load_dataset import importKaggleOld, loadFromDataFrame
from preprocessing import Preprocessing
from neural_network_utils import *
from clustering_algorithms import TSNEAlgo, PCAAlgo, KMeansCluster, SpectralCluster



#CSV load files
KAGGLE_TRAIN = 'csvloadfiles/kaggle_original.csv'
KAGGLE_TEST = 'csvloadfiles/kaggle_five_classes.csv'
KAGGLE_MISSING = 'csvloadfiles/kaggle_missing.csv'
KAGGLE_MISSING_TEST = 'csvloadfiles/kaggle_missing_five.csv'



class RotationInvariantAutoencoder():
    def __init__(self):

            return


    def autoencoder_architecture(self, input_shape = (64,64,3),output_shape = 10, network= None):

        # define the input to the encoder
        inputShape = input_shape
        inputs = keras.Input(shape=inputShape)

        gconv =  GConv2D(64, (3, 3), kernel_initializer='he_normal',
                            padding="same", name='Gconv2D_0',
                            strides=(1, 1), use_bias=False, kernel_regularizer=None,
                            h_input='Z2', h_output='C4')(inputs)
        x = GBatchNorm('C4')(gconv)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

        gconv =  GConv2D(128, (3, 3), kernel_initializer='he_normal',
                            padding="same", name='Gconv2D_1',
                            strides=(1, 1), use_bias=False, kernel_regularizer=None,
                            h_input='C4', h_output='C4')(x)
        x = GBatchNorm('C4')(gconv)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

        gconv =  GConv2D(256, (3, 3), kernel_initializer='he_normal',
                            padding="same", name='Gconv2D_2',
                            strides=(1, 1), use_bias=False, kernel_regularizer=None,
                            h_input='C4', h_output='C4')(x)
        x = GBatchNorm('C4')(gconv)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

        gconv =  GConv2D(512, (3, 3), kernel_initializer='he_normal',
                            padding="same", name='Gconv2D_3',
                            strides=(1, 1), use_bias=False, kernel_regularizer=None,
                            h_input='C4', h_output='C4')(x)
        x = GBatchNorm('C4')(gconv)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

        gconv =  GConv2D(512, (3, 3), kernel_initializer='he_normal',
                            padding="same", name='Gconv2D_4',
                            strides=(1, 1), use_bias=False, kernel_regularizer=None,
                            h_input='C4', h_output='C4')(x)
        x = GBatchNorm('C4')(gconv)
        x = keras.layers.LeakyReLU()(x)

        ### latent dim ###
        group_pool = GroupPool(h_input='C4',name = 'group_pool')(x)

        #GAP LAYER
        encoder_output = keras.layers.GlobalAveragePooling2D()(group_pool)

        encoder = keras.Model(inputs, encoder_output, name="Gconv")

        ### Decoder ####

        #Get outputsize of last convlayer
        volumeSize = K.int_shape(x)

        # Dense layer + reshape to get feature map shape
        latentInputs = keras.layers.Input(shape=(512,))
        x = keras.layers.Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = keras.layers.Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)

        gconv =  keras.layers.Conv2DTranspose(512, (3,3), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(gconv)
        x = keras.layers.LeakyReLU()(x)

        gconv =  keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(gconv)
        x = keras.layers.LeakyReLU()(x)

        gconv =  keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(gconv)
        x = keras.layers.LeakyReLU()(x)

        gconv =  keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(gconv)
        x = keras.layers.LeakyReLU()(x)

        gconv =  keras.layers.Conv2DTranspose(64, (3,3), strides=(2, 2), padding='same')(x)
        x = keras.layers.BatchNormalization()(gconv)
        x = keras.layers.LeakyReLU()(x)

        #x = GroupPool(h_input='C4')(x)
        outputs = keras.layers.Conv2DTranspose(3,(3,3),kernel_initializer='he_normal', activation = 'tanh', padding="same")(x)

        # build the decoder model
        decoder = keras.Model(latentInputs, outputs, name="decoder")

        # Build Model autoencoder
        autoencoder = keras.Model(inputs, decoder(encoder(inputs)),name="autoencoder")

        return autoencoder, encoder, decoder

    def plot(self,model, data):

        ix = randint(0, data.shape[0], 16)
    	# select images
        data = data[ix]

    	# prepare fake examples
        X = model.predict(data)

        fig = plt.figure(figsize=(4,4))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(X[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('autoencoder.png')
        plt.close()


    def train(self,name = ''):
        autoencoder, encoder, decoder = self.autoencoder_architecture(network = 'autoencoder')
        print(encoder.summary())
        print(decoder.summary())
        print(autoencoder.summary())

        optimizer = keras.optimizers.Adam()
        autoencoder.compile(loss='mse', optimizer=optimizer)
        print('Finished compiling')

        #train_images, train_images = importKaggleOld()
        train_images,train_labels = loadFromDataFrame(KAGGLE_TRAIN)
        X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

        pre_train = Preprocessing(X_train,y_train,autoencoder = True)
        pre_validate = Preprocessing(X_test,y_test,autoencoder = True)
        generator = pre_train.returnAugmentedDataset()
        validation_data = pre_validate.returnDataset()

        # Test equivariance by comparing outputs for rotated versions of same datapoint:
        res = encoder.predict(np.stack([validation_data[123], np.rot90(validation_data[123])]))
        print(res.shape)

        is_equivariant = np.allclose(res[0][0], res[1][0])
        print('Equivariance check:', is_equivariant)

        print("Training AutoEncoder")
        autoencoder.fit_generator(generator,steps_per_epoch= pre_train.shape()[0] / 32,validation_data=(validation_data, validation_data),epochs=100)

        print("plotting reconstruction")
        self.plot(autoencoder, validation_data)

        print('saving weights')
        saveWeights(encoder,str(savename))

    def test(self,name = ''):

        # Load images
        train_images,train_labels = loadFromDataFrame(KAGGLE_TEST)
        images_black_and_white = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (train_images)])
        test_set = Preprocessing(train_images,train_labels,autoencoder = True).returnImages()

        autoencoder, encoder, decoder = self.autoencoder_architecture(network = 'autoencoder')
        loadWeights(encoder,str(0))

        def getRandom(data):
            list = [16403,4953,29872,7892,23613,1747,19272,17585,21178]
            img = []
            #for i in range(16):
            #    list.append(randint(0,30000))
            for i, val in enumerate(list):
                img.append(data[val])
            print(list)
            return np.array(img)

        #visualize_class_activation_map(encoder, getRandom(train_images))

        features = encoder.predict(test_set)




        #Predict using K-means
        print("KMEANS CLUSTERING")
        kmean = KMeansCluster(n_clusters = 5)
        kmean.fit(features)
        k_means_labels = kmean.sortLabels(train_labels,kmean.predict(features))
        accuracy, precision, recall, f1score = kmean.performance(train_labels,k_means_labels)

        print("Spectral CLUSTERING")
        spectral = SpectralCluster(n_clusters = 5)
        spectral.fit(features)
        spectral_labels = spectral.sortLabels(train_labels,spectral.predict(features))
        accuracy, precision, recall, f1score = spectral.performance(train_labels,spectral_labels)

        #Visualize using TSNE
        TSNE = TSNEAlgo()
        TSNE.tsne_fit(features,perplexity = 35)


        #TSNE.tsne_plot(images_black_and_white,train_labels,"baseline","baseline")
        #TSNE.tsne_plot(images_black_and_white,k_means_labels,"kmeans","baseline")
        #TSNE.tsne_plot(images_black_and_white,spectral_labels,"spectral","baseline")
