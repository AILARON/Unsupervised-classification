#########################################################
#                    Main function                      #
#                                                       #
#Created by Eivind Salvesen                             #
#########################################################

import os
import numpy as np
import cv2
import tensorflow as tf

from auto_encoder import AutoEncoder
from load_dataset import LoadDataset
from augment_data import DataAugmentation
from network_model import NetworkModel
from results import getNetworkResults

#Set up GPU
gpu = 1
if gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

if gpu == 1:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')

    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    #tf.config.experimental.set_memory_growth(physical_devices[1], True)


def dataAugment():
    """
    Augment data by fetching a load dir and saving augmented data in save dir
    """
    load_dir = "dataset/trainoriginalfull/"
    save_dir = "dataset/kaggle_augmented_train_new/"
    augment = DataAugmentation(load_dir,save_dir)
    augment.dataAugmentation()


def buildNetwork(model_type,latent_vector,latent_dim = 10, epochs = 10,train = True,noisy=False):
    """
    Builds auto_encoder network
    Input: 1. model_type can be "FULLYCONNECTED", "VGG" or "COAPNET"
    Input: 2. latent_vector can be "dense", "globalAverage" or "sparse"
    Input: 3. latent_dim size of preffered bottleneck
    Input: 4. epochs: number of training epochs
    Input: 5. train: true/false if false fetch model from savefile
    Input: 6. noisy: true/false train on noisy data or not
    """
    # build and train regularized model
    if noisy:
        model = RegularizedNetworkModel("dataset/trainoriginalfull/",
        latent_vector = latent_vector,latent_dim = latent_dim, model_type=model_type, num_epochs = epochs)
    else:
        model = NetworkModel("dataset/trainoriginalfull/",
        latent_vector = latent_vector,latent_dim = latent_dim, model_type=model_type, num_epochs = epochs)
    #model = NetworkModel("dataset/kaggle_augmented_train",conv_layers = [256,128,64],
    #latent_vector = 32, model_type=model_type, num_epochs = 50)
    if train == True:
        model.runModel()
        model.saveWeights()
    else:
        model.buildModel()
        model.loadWeights()

    return model


if __name__=="__main__":
    tf.keras.backend.clear_session()

    model = "auto"


    if model == "auto":
        model_type = "COAPNET"
        latent_vector = "globalAverage"
        model = buildNetwork(model_type, latent_vector,latent_dim = 64, epochs = 50,train = True,noisy = False)
        auto, enc, pre = model.getModel()
        getNetworkResults(model,latent_vector)


    if model == "dec":
        train_data = LoadDataset("dataset/kaggle_original_train/",0)
        train_data, train_label, val, val_label = train_data.load_data()
        encoded_x= np.reshape(train_data,(3720, 64 *64))

        model_type = "COAPNET"
        latent_vector = "globalAverage"
        model = buildNetwork(model_type, latent_vector,latent_dim = 64, epochs = 5,train = True,noisy = False)
        auto, enc, pre = model.getModel()


        from DEC import DeepEmbeddedClustering
        from results import getDECNetworkResults

        dec = DeepEmbeddedClustering(auto,enc,train_data,train_label,5)
        dec.buildModel()
        dec.trainModel()
        dec,enc = dec.getModel()
        getDECNetworkResults(dec,enc)

    if model == "vae":
        train_data = LoadDataset("dataset/trainoriginalfull/",0.1)
        train_data, test_label, val_data, val_label = train_data.load_data()
        from variational_auto_encoder import vae

        model = vae(train_data,val_data)

        """
        #Get results from vae
        train_data = LoadDataset("dataset/kaggle_original_train/",0)
        train_data, train_label, val, val_label = train_data.load_data()
        x  = train_data.reshape(train_data.shape[0], 64, 64, 1).astype('float32')

        ds = np.zeros((train_data.shape[0],64))
        i = 0
        train_dataset = tf.data.Dataset.from_tensor_slices(x).batch(32)
        for train_x in train_dataset:
            data,_ = model.encode(train_x)
            for enc in data:
                ds[i] = enc
                i = i +1

        TSNE = TSNEAlgo()
        TSNE.tsne_fit(ds,perplexity = 35)

        kmean = KMeansClustering()
        kmean.fit(ds)
        predKM  = kmean.predict(ds)
        accuracy(train_label,predKM)

        model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',
                                   assign_labels='kmeans')
        predSPEC = model.fit_predict(ds)
        accuracy(train_label,predSPEC)

        train_x = np.reshape(train_data,(3720,64,64))
        TSNE.tsne_plot(train_x,predSPEC,save_data_dir ="vae",save_name="spectral")
        TSNE.tsne_plot(train_x,train_label,save_data_dir ="vae",save_name="true_label")
        TSNE.tsne_plot(train_x,predKM,save_data_dir ="vae",save_name="kmean")
        """

    ############################################################################
