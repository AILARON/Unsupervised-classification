#################
#Network class builds and train a network using the networks defined in the AutoEncoder class
#################

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, SGD

from auto_encoder import AutoEncoder
from sparse_autoencoder import SparseAutoEncoder
from load_dataset import LoadDataset
from augment_data import DataAugmentation
from cbof import initialize_bof_layers

class NetworkModel:
    load_data_dir = None

    #Dataset:
    train_data = None
    train_label = None
    validation_data = None
    validation_label = None

    #model
    auto_encoder = None
    encoder = None
    decoder = None

    #model specific variables
    conv_layers = None
    latent_vector = None
    model_type = None
    latent_vector = None

    num_epochs = 50

    def __init__(self,load_data_dir,model_type,latent_vector = "dense",latent_dim = 10,num_epochs = 50):
        self.load_data_dir = load_data_dir
        self.latent_vector = latent_vector
        self.model_type = model_type
        self.num_epochs = num_epochs
        self.latent_vector = latent_vector
        self.latent_dim = latent_dim

    def loadData(self):
        train_data = LoadDataset(self.load_data_dir,0.1)
        self.train_data, self.train_label,self.validation_data, self.validation_label = train_data.load_data()
        return

    def buildModel(self):
        """
        Builds autoencoder models based on init parameters
        """

        if self.model_type == "VGG":
            autoencoder = AutoEncoder(model_type ="VGG",latent_vector = self.latent_vector, latent_dim = self.latent_dim)
            # construct the convolutional autoencoder
            self.auto_encoder, self.encoder, self.decoder  = autoencoder.VGG()

        elif self.model_type == "COAPNET":
            autoencoder = AutoEncoder(model_type ="COAPNET",latent_vector = self.latent_vector, latent_dim = self.latent_dim)
            # construct the convolutional autoencoder
            self.auto_encoder, self.encoder, self.decoder  = autoencoder.COAPNET()

        elif self.model_type == "FULLYCONNECTED":
            autoencoder = AutoEncoder(model_type ="FULLYCONNECTED",latent_vector = self.latent_vector, latent_dim = self.latent_dim)
            # construct the convolutional autoencoder
            self.auto_encoder, self.encoder, self.decoder  = autoencoder.FULLYCONNECTED()

        elif self.model_type == "vae":
            from test import vae_test

            self.auto_encoder, self.encoder, self.decoder = vae_test()
            print(self.encoder.summary())
            print(self.decoder.summary())
            print(self.auto_encoder.summary())
            # construct the convolutional autoencoder
            #self.auto_encoder, self.encoder, self.decoder  = autoencoder.sparse()
            return

        else:
            print("No model with that name exists")


        adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.01, nesterov=False)
        self.auto_encoder.compile(loss="mse", optimizer=adam) #SGD(lr=1e-3)
        print(self.encoder.summary())
        print(self.decoder.summary())
        print(self.auto_encoder.summary())


    def train(self,num_epochs):
        if self.model_type == "vae":
            history = self.auto_encoder.fit(x = self.train_data,y = self.train_data,batch_size = 32,
            validation_data = (self.validation_data,self.validation_data),shuffle=True,
            epochs = 10,verbose= 1)
            
        elif self.model_type == "FULLYCONNECTED":
            train_data= np.reshape(self.train_data,newshape  = (self.train_data.shape[0],64*64*1))
            val_data = np.reshape(self.validation_data,newshape  = (self.validation_data.shape[0],64*64*1))
            import math
            stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='auto',
            baseline=None, restore_best_weights=False)

            history = self.auto_encoder.fit(x = train_data,y = train_data,batch_size = 32,
            steps_per_epoch=int(math.ceil(1. * self.train_data.shape[0] / 32)),validation_steps=int(math.ceil(1. * self.validation_data.shape[0] / 32)),
            validation_data = (val_data,val_data),shuffle=True,
            epochs = num_epochs,verbose= 1, callbacks = [stopping])

        else:
            import math
            stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='auto',
            baseline=None, restore_best_weights=False)

            history = self.auto_encoder.fit(x = self.train_data,y = self.train_data,batch_size = 32,
            steps_per_epoch=int(math.ceil(1. * self.train_data.shape[0] / 32)),validation_steps=int(math.ceil(1. * self.validation_data.shape[0] / 32)),
            validation_data = (self.validation_data,self.validation_data),callbacks = [stopping],shuffle=True,
            epochs = num_epochs,verbose= 1)

        return history

    def saveWeights(self):
        if self.model_type == "COAPNET":
            # Save JSON config to disk
            json_config = self.auto_encoder.to_json()
            with open('model_config.json', 'w') as json_file:
                json_file.write(json_config)
                # Save weights to disk
                print("[Info] saving COAPNET weights")
                self.auto_encoder.save_weights(str(self.model_type)+"_"+str(self.latent_vector)+'_weights.h5')

        elif self.model_type == "VGG":
            # Save JSON config to disk
            json_config = self.auto_encoder.to_json()
            with open('model_config.json', 'w') as json_file:
                json_file.write(json_config)
                # Save weights to disk
                print("[Info] saving VGG weights")
                self.auto_encoder.save_weights(str(self.model_type)+"_"+str(self.latent_vector)+'_weights.h5')

        elif self.model_type == "fullyConnected":
            # Save JSON config to disk
            json_config = self.auto_encoder.to_json()
            with open('model_config.json', 'w') as json_file:
                json_file.write(json_config)
                # Save weights to disk
                print("[Info] saving fullyConnected weights")
                self.auto_encoder.save_weights('fullyConnected_weights.h5')
        else:
            print("No model name corresponding")

    def loadWeights(self):
        print("[Info] loading previous weights")
        try:
            if self.model_type == "COAPNET":
                self.auto_encoder.load_weights(str(self.model_type)+"_"+str(self.latent_vector)+'_weights.h5')
            elif self.model_type == "VGG":
                self.auto_encoder.load_weights(str(self.model_type)+"_"+str(self.latent_vector)+'_weights.h5')
            elif self.model_type == "fullyConnected":
                self.auto_encoder.load_weights('fullyConnected_weights.h5')
        except:
            print("Could not load weights")

    def printHistory(self,history,modelname):
        """
        Plots the training and validation accuracy for a network
        :param history: the accuracy and loss of the given model for every epoch
        :param modelname: the name of the network used for training
        """
        # summarize history for accuracy
        try:
            plt.figure()
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title(modelname +' accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('acc.png')
        except:
            print("No accuracy in history block")
        try:
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(modelname +' loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(modelname+'_'+'loss.png')
        except:
            print("No loss in history block")

    def runModel(self):
        self.loadData()
        self.buildModel()
        history = self.train(self.num_epochs)
        self.printHistory(history,self.model_type)


    def getModel(self):
        return self.auto_encoder, self.encoder, self.decoder

    def getPretrainedModel(self):
        self.loadWeights()
        return self.auto_encoder, self.encoder, self.decoder
