import os
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split

import cv2

from tensorflow.keras import layers

import numpy as np
from load_dataset import importKaggle, importAilaron
from clustering_algorithms import TSNEAlgo, PCAAlgo

from clustering_algorithms import KMeansCluster, SpectralCluster
from utils import accuracy
from preprocessing import Preprocessing
from deep_neural_networks import VGG_BATCHNORM

def network():

    #load training data
    train_data, train_labels = importKaggle()
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=121, dtype='float32')

    print(train_labels.shape)
    print(train_labels)

    # Split data
    train_data, test_data, train_label, test_label = train_test_split(train_data, train_labels, test_size=0.1)

    for i in range(20):
        train(i, train_data, test_data, train_label, test_label)

def train(i, train_data, test_data, train_label, test_label):
    shape = (160+20*i,160+20*i,3)
    model = VGG_BATCHNORM(input_shape=shape,output_shape = 121)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    # Preprocess data
    train_dataset = Preprocessing(train_data, train_label,input_shape = shape).returnAugmentedDataset()
    test_dataset = Preprocessing(test_data, test_label,input_shape = shape).returnDataset()

    #Early stopping criterion
    stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=30, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False)

    history = model.fit(train_dataset,validation_data =test_dataset, steps_per_epoch= train_data.shape[0] // 32,
    validation_steps= test_data.shape[0] // 32, verbose = 0, callbacks =[stopping], epochs = 30)

    print(160+20*i,':   ',history.history['loss'][29],history.history['val_loss'][29],history.history['accuracy'][29],history.history['val_accuracy'][29])
    return
