######################
#Helper functions for printing bag of features codewords for same class samples
######################

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os


def getSameSamples(x_train,y_train,class_name):
    """
    Function returning same class objects.
    Input: 1. data set
    Input: 2. labels
    Input: 3. label 0-n that one would like to return
    Output: data and labels all from same class
    """
    #y_train = tf.keras.utils.to_categorical(y_train, num_classes=5, dtype='bool')
    print(y_train)
    train_mask = np.isin(y_train,[class_name])
    print(train_mask)
    x_train, y_train = x_train[train_mask], y_train[train_mask]

    print(x_train.shape)
    return x_train, y_train

def makeSamplevec(x_pred):
    """
    Concatinating input vectors to create one average vector output.
    Input:  Data set
    Output: Average vector of all input samples
    """
    vec=[[0 for i in range(64)]]
    for i in range(len(x_pred)):
        vec += x_pred[i]
    vec_averaged = vec/len(x_pred)
    print(vec_averaged)
    return vec_averaged[0]

def printVec(x_one,x_two,x_four):
    """
    Printing three sample vectors in the same figure
    Input:  Sample vector 1,2,3
    """
    t = np.arange(0.0, 64.0, 1)
    plt.figure(1)
    plt.subplots(figsize = (20,8))
    print("--------- wordbooks for number 1, 2 and 4 ----------")
    plt.subplot(411)
    plt.xticks(t)
    plt.plot(t, x_one,"b")
    blue_patch = mpatches.Patch(color='blue', label='Number 1')
    plt.legend(handles=[blue_patch])

    plt.subplot(412)
    plt.xticks(t)
    plt.plot(t, x_two,"r")
    red_patch = mpatches.Patch(color='red', label='Number 3')
    plt.legend(handles=[red_patch])

    #t = np.arange(0.0, 128.0, 1)
    plt.figure(2)
    plt.subplot(413)
    plt.xticks(t)
    plt.plot(t, x_four,"g")
    green_patch = mpatches.Patch(color='green', label='Number 8')
    plt.legend(handles=[green_patch])

    plt.subplot(414)
    plt.xticks(t)
    plt.plot(t, x_one,"b",t, x_two,"r",t, x_four,"g")

    plt.savefig(os.path.join('_image.png'))
    return 0
