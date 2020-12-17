##################
#Keras based autoencoder implementation of symmetric encoder-decoder
#Using VGG16 based, fullyConnected and COAPNET as feature extractor

#This file uses ideas from these sources in its implementation
#https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/
#https://blog.keras.io/building-autoencoders-in-keras.html

"""
Input: 1. model_type: which encoder to use: fullyConnected network, VGG16 or COAPNET
Input: 2. latent_vector: Which type of latent vector to use: dense, globalAverage or sparse
Input: 3. latent_dim: Latent dimension to use
"""
##################


import tensorflow as tf
import numpy as np
from cbof import BoF_Pooling
from regularizer import SparseActivityRegularizer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
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


class AutoEncoder:
    width = 64
    height = 64
    depth = 1
    latent_dim = 10
    model_type = "VGG"
    latent_vector = "dense"

    def __init__(self, model_type = "VGG", latent_vector = "dense", latent_dim = "10"):
        self.model_type = model_type
        self.latent_vector = latent_vector
        self.latent_dim = latent_dim
        return

    def VGG(self):
        """
        Convolutional auto-encoder model, symmetric.
        Using the cnn network implementation VGG16 as feature extractor
        """
        # define the input to the encoder
        inputShape = (self.height, self.width, self.depth)
        inputs = Input(shape=inputShape)

        # apply 2 x (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(64, (3, 3), padding="same" )(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2, 2), padding='same')(x)


        # apply 2 x (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(128, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2, 2), padding='same')(x)


        # apply 3 x (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(256, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2, 2), padding='same')(x)


        # apply 3 x (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(512, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2, 2), padding='same')(x)


        #Add Latent layer
        if self.latent_vector == "globalAverage":
            x = Conv2D(self.latent_dim, (3, 3),padding="same", activation = "relu", name = "encoded5" )(x)

            #Get outputsize of last convlayer
            volumeSize = K.int_shape(x)

            # flatten the network and then construct our latent vector
            latent = GlobalAveragePooling2D()(x)

            ####### build the Encoder model ########
            encoder = Model(inputs, latent, name="encoder")

            #Building decoder
            latentInputs = Input(shape=(self.latent_dim,))
            x = Dense(np.prod(volumeSize[1:]))(latentInputs)
            x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)

            x = Conv2D(self.latent_dim, (3, 3),padding="same", activation = "relu")(x)


        elif self.latent_vector == "sparse":
            #Get outputsize of last convlayer
            volumeSize = K.int_shape(x)

            # flatten the network and then construct our latent vector
            x = Flatten()(x)
            regularizer = SparseActivityRegularizer(0.001, 10)
            #tf.keras.regularizers.l1(10e-4)
            latent = Dense(self.latent_dim, activity_regularizer=regularizer,activation='sigmoid' )(x)

            ####### build the Encoder model ########
            encoder = Model(inputs, latent, name="encoder")

            #Building decoder
            latentInputs = Input(shape=(self.latent_dim,))
            x = Dense(np.prod(volumeSize[1:]))(latentInputs)
            x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)


        elif self.latent_vector == "dense":
            #Get outputsize of last convlayer
            volumeSize = K.int_shape(x)

            # flatten the network and then construct our latent vector
            x = Flatten()(x)
            latent = Dense(self.latent_dim,activation ="relu")(x)

            ####### build the Encoder model ########
            encoder = Model(inputs, latent, name="encoder")

            #Building decoder
            latentInputs = Input(shape=(self.latent_dim,))
            x = Dense(np.prod(volumeSize[1:]))(latentInputs)
            x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)



        # apply UpSampling + 3 x (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(512, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)


        # apply UpSampling + 3 x (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(256, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)


        # apply UpSampling + 2 x (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(128, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)


        # apply UpSampling + 2 x (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(64, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)


        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        outputs = Conv2D(self.depth, (3, 3), padding="same", activation = "sigmoid")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # Build Model autoencoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),name="autoencoder")

        return autoencoder, encoder, decoder

    def COAPNET(self):
        """
        Convolutional auto-encoder model, symmetric.
        Using the cnn network implementation COAPNET as feature extractor
        """
        # define the input to the encoder
        inputShape = (self.height, self.width, self.depth)
        inputs = Input(shape=inputShape)

        # apply (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(64, (3, 3), padding="same" )(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)


        # apply (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(128, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)


        # apply (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(256, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)


        # apply (CONV => BN layer => ReLU activation) + MaxPooling
        x = Conv2D(512, (3, 3), padding="same" )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)


        #Add Latent layer
        if self.latent_vector == "globalAverage":
            x = Conv2D(self.latent_dim, (3, 3),padding="same", activation = "relu", name = "encoded5" )(x)

            #Get outputsize of last convlayer
            volumeSize = K.int_shape(x)

            # flatten the network and then construct our latent vector
            latent = GlobalAveragePooling2D()(x)

            ####### build the Encoder model ########
            encoder = Model(inputs, latent, name="encoder")

            #Building decoder
            latentInputs = Input(shape=(self.latent_dim,))
            x = Dense(np.prod(volumeSize[1:]))(latentInputs)
            x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)

            x = Conv2D(self.latent_dim, (3, 3),padding="same", activation = "relu")(x)


        elif self.latent_vector == "sparse":
            #Get outputsize of last convlayer
            volumeSize = K.int_shape(x)

            # flatten the network and then construct our latent vector
            x = Flatten()(x)
            regularizer = SparseActivityRegularizer(0.001, 20)
            latent = Dense(self.latent_dim, activity_regularizer=regularizer,activation='sigmoid' )(x)

            ####### build the Encoder model ########
            encoder = Model(inputs, latent, name="encoder")

            #Building decoder
            latentInputs = Input(shape=(self.latent_dim,))
            x = Dense(np.prod(volumeSize[1:]))(latentInputs)
            x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)


        elif self.latent_vector == "dense":
            #Get outputsize of last convlayer
            volumeSize = K.int_shape(x)

            # flatten the network and then construct our latent vector
            x = Flatten()(x)
            latent = Dense(self.latent_dim,activation ="relu")(x)

            ####### build the Encoder model ########
            encoder = Model(inputs, latent, name="encoder")

            #Building decoder
            latentInputs = Input(shape=(self.latent_dim,))
            x = Dense(np.prod(volumeSize[1:]))(latentInputs)
            x = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)



        # apply UpSampling + (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(512, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        # apply UpSampling + (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(256, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        # apply UpSampling + 2 x (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        # apply UpSampling + (CONV => BN layer => ReLU activation)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3),padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        outputs = Conv2D(self.depth, (3, 3), padding="same", activation = "sigmoid")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # Build Model autoencoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),name="autoencoder")

        return autoencoder, encoder, decoder

    def FULLYCONNECTED(self):
        """
        Fully connected auto-encoder model, symmetric.
        Copy of the https://github.com/XifengGuo/DEC-keras implementation used as reference
        """

        # input
        width = 64
        height = 64
        depth = 1
        latent_dim = self.latent_dim

        # define the input to the encoder
        inputShape = (height *width* depth)
        inputs = Input(shape=inputShape)
        # internal layers in encoder

        x = Dense(500, activation="relu")(inputs)
        x = Dense(500, activation="relu")(x)
        x = Dense(500, activation="relu")(x)
        x = Dense(2000, activation="relu")(x)
        # hidden layer
        latent = Dense(latent_dim, activation="relu")(x)  # hidden layer, features are extracted from here

        ####### build the Encoder model ########
        encoder = Model(inputs, latent, name="encoder")

        #Building decoder
        latentInputs = Input(shape=(latent_dim,))
        x = Dense(2000, activation="relu")(latentInputs)
        x = Dense(500, activation="relu")(x)
        x = Dense(500, activation="relu")(x)
        outputs = Dense(inputShape, activation="relu")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # Build Model autoencoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),name="autoencoder")

        return autoencoder, encoder, decoder
