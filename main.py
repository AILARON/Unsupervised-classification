#########################################################
#                    Main function                      #
#                                                       #
#               Created by Eivind Salvesen              #
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

from deep_cluster import DeepCluster
#from deepclusterold import DeepCluster

#Set up GPU
gpu = 1
if gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

if gpu == 1:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.experimental.set_memory_growth(physical_devices[1], True)


if __name__=="__main__":
    tf.keras.backend.clear_session()

    from load_dataset import createCSVFile, checkCSVFile

    #createCSVFile('kaggle_five_classes','kaggle_five_classes_missing',drop_class = True)
    #checkCSVFile(kaggle_missing)
    #os.exit()

    from gradcam import grad
    from bof_model import example

    from gan import GAN
    from cgan import cGan
    from mnistgan import gan
    from preprocessing import makeCSVFile
    from baseline import neuralNetwork
    from traditional_extraction import traditionalExtraction
    from get_dataset_info import data_info
    from size_experiment import network

    #grad()
    #sys.exit()

    model = "DC"

    if model == "rotation_invariant":
        from rot_inv_autoencoder import RotationInvariantAutoencoder
        auto = RotationInvariantAutoencoder()
        for i in range(5):
            auto.train(name = i)
            auto.test(name = i)


    if model == "data_statistics":

        data_info().data_statistics()

    if model == "traditionalExtraction":
        traditionalExtraction()

    if model == "neuralNetwork":
        neuralNetwork(1)

    if model == "network":
        network()

    if model == "DC":
        for i in range(5):
            DeepCluster().train(1, initialize_previous = False,name = i)

    if model == "gan":
        model = GAN()
        model.train_gan()
        model.test_gan()

    ############################################################################
