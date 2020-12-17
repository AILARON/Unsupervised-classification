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
    tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')

    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    #tf.config.experimental.set_memory_growth(physical_devices[1], True)

print(tf.__version__)

if __name__=="__main__":
    tf.keras.backend.clear_session()

    from load_dataset import createCSVFile, checkCSVFile

    #createCSVFile('ailaron_small','ailaron_test',drop_class = False)
    #checkCSVFile('ailaron_original')
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
    from classification import ClusterModels
    #grad()
    #sys.exit()

    model = "classification"

    if model == "classification":
        print(tf.__version__)
        #if tf.__version__ == '1.15.0':
        #    ClusterModels().dec(model = 'auto',train = False)

        #else:
        #    ClusterModels().dec(model = 'DC',train = False)

        print(tf.__version__)
        if tf.__version__ == '1.15.0':
            from rot_inv_autoencoder import RotationInvariantAutoencoder
            ClusterModels().testAuto()
        else:
            ClusterModels().testDC()

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
        #for i in range(3):
        #    neuralNetwork(i)

    if model == "network":
        network()

    if model == "DC":
        for i in range(5):
        #from test import DeepCluster
            DeepCluster().train(1, initialize_previous = False,name = i)

    if model == "gan":
        model = GAN()
        model.train_gan()
        model.test_gan()

    ############################################################################
