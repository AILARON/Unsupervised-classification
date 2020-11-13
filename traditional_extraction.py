#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Author: Aya Saad
# Date created: 24 September 2019
# Project: AILARON
# funded by RCN FRINATEK IKTPLUSS program (project number 262701) and supported by NTNU AMOS
#
#################################################################################################################


######
#RUN USING SIFTSURFENV
######

import numpy as np
from descriptors import Descriptor, BFMatcher

from clustering_algorithms import *

import tensorflow as tf
import scipy.spatial.distance as ssd
from load_dataset import importKaggle, importAilaron
import cv2

def calculate_distances(input_data, desc ='SIFT', image_type = "rgb"):
    desc_len = 64       # SURF descriptor length
    if desc == 'SIFT':
        desc_len = 128

    X_desc = []
    temp = []
    for image in input_data: #.iloc[:, 0]:
        #print(image)
        #image = Image(x).image_read(resize=False)


        #img = np.float64(image) / np.max(image)
        #img = image.astype(np.float32)

        #reshape
        if image_type == "rgb":
            img = image.reshape(64,64,3)
        elif image_type == "bw":
            img = image.reshape(64,64)

        #print(img.dtype)
        #print(img.shape)

        #temp.append(img)
        img_, _, fd = Descriptor(desc, img).algorithm(desc)
        X_desc = np.append(X_desc, None)  # Allocate space for your new array
        X_desc[-1] = fd
        temp.append(img_)

    dist = []
    print('len(X_desc)',len(X_desc))
    for i in range(0, len(X_desc)):
        for j in range(i + 1, len(X_desc)):
            sim = -1
            if ((X_desc[i] is not None) and (X_desc[j] is not None)):
                sim = BFMatcher().match_images(desc_a=X_desc[i], desc_b=X_desc[j])
            dist.append(sim)

    for i in range(0, len(X_desc)):
        if X_desc[i] is None:
            X_desc[i] = np.zeros((2, 3), dtype=float)
    print('max(X) = ', max(dist))
    max_value = max(dist)
    dist_desc = [x if x >= 0 else max_value for x in dist]

    ####################################################
    X_I = np.zeros([len(X_desc), len(max(X_desc, key=lambda x: len(x))), desc_len])

    for i, j in enumerate(X_desc):
        for k, l in enumerate(j):
            X_I[i][k][0:len(l)] = l
        print(i, len(j), X_I[i][0:len(j)], j, end='  ')

    x_size = len(input_data)
    X_I = X_I.reshape(x_size, -1).astype('float32')
    print('X_I.shape  ', X_I.shape)

    ####################################################

    return X_I, dist_desc, temp

def calculate_hog_distances(input_data, desc ='HOG'):
    data = []
    temp = []
    for x in input_data.iloc[:, 0]:
        print(x)
        image = Image(x).image_read(resize=True)
        img = np.float64(image) / np.max(image)
        img = img.astype('float32')
        temp.append(img)
        _, _, fd = Descriptor('HOG', image).algorithm('HOG')
        data.append(fd)
    X = np.stack(temp)
    X /= 255.0
    x_size = len(input_data)
    X = X.reshape(x_size, -1).astype('float32')

    X_hog = np.stack(data)
    X_hog = X_hog.reshape(x_size, -1).astype('float32')

    print(X.shape)
    return X, X_hog


def traditionalExtraction(exct = 1):
    extractor_types = ["SIFT","SURF","HOG"]
    extractor_method = extractor_types[exct]

    #Import data
    #input_data, input_labels = importKaggleTest(depth=1)
    input_data, input_labels = importAilaronTest(depth=1)
    print(input_labels)
    #Resize images
    #input_data = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (input_data)], dtype=np.uint8) #, dtype=np.uint8 SURF

    #resize and normalize
    input_data = np.array([cv2.resize(cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX), dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (input_data)], dtype=np.uint8) #, dtype=np.uint8 SURF


    input_data_center = input_data - input_data.mean()

    #input_data = input_data[:1000,:,:]


    if extractor_method == "SIFT":
        X_I, X_SIFT, img = calculate_distances(input_data, desc='SIFT', image_type = "rgb")

        #Visualize sift key points
        printImages(img,name=extractor_method)

        #Calculate and print cluster results
        clusterResults(extractor_method,X_I, X_SIFT, input_labels)

        labels_sift = \
            HierarchicalClustering() \
                    .draw_dendogram(X_SIFT,
                                title='Hierarchical Clustering Dendrogram of the SIFT Descriptors')


    elif extractor_method == "SURF":
        X_U, X_SURF, img = calculate_distances(input_data, desc='SURF', image_type = "rgb")

        #Visualize sift key points
        printImages(img,name=extractor_method)

        #Calculate and print cluster results
        clusterResults(extractor_method,X_U, X_SURF, input_labels)

        labels_sift = \
            HierarchicalClustering() \
                    .draw_dendogram(X_SURF,
                                title='Hierarchical Clustering Dendrogram of the SIFT Descriptors')

    #Get features from deep neural network for better visualization
    features = extractDeepFeatures(input_data_center)

    #Make TSNE plot using sift descriptors and predicted sift labels
    tsne = TSNEAlgo()
    tsne.tsne_fit(X_U,perplexity=35)
    tsne.tsne_plot(input_data, labels_sift, extractor_method,extractor_method)

    #Make TSNE plots using features from deep network, predicted sift labels and true labels
    tsne = TSNEAlgo()
    tsne.tsne_fit(features,perplexity=35)
    tsne.tsne_plot(input_data, input_labels, extractor_method+"_COAP_TRUE",extractor_method)
    tsne.tsne_plot(input_data, labels_sift,extractor_method+"_COAP_PRED",extractor_method)

    # Get image descriptor
    #X_I, X_SIFT, img = calculate_distances(input_data, desc='SIFT')
    #X_U, X_SURF = calculate_distances(input_data, desc='SURF')
    #X, X_hog = calculate_hog_distances(input_data, desc='HOG')
    print('------------------------------------------')

    return

from deep_neural_networks import COAPNET
from baseline import loadWeights
def extractDeepFeatures(input_data):
    #Import pretrained Coapnet classifier and compile
    model = COAPNET()
    model = loadWeights(model)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.01, nesterov=False)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    #Make dataset 3 dimensional
    #train_data_3d=np.stack([input_data]*3, axis=-1)
    #train_data_3d = train_data_3d.reshape(-1, 64,64,3)

    #Make tensorflow dataset
    visualize_dataset = tf.data.Dataset.from_tensor_slices(input_data)
    visualize_dataset = visualize_dataset.batch(32)

    #Remove softmax layer
    output = model.layers[-2].output
    model_extractor = tf.keras.models.Model(inputs=model.input,
                       outputs=output)
    #Extract features
    features = model_extractor.predict(visualize_dataset)

    return features

import random


def printImages(image_data, name ="name"):
    hstack = None
    vstack = None
    for i in range (4):
        hstack = None
        for j in range (4):

            original = image_data[random.randint(0, 7000)] #*255#image_data[i*4 +j]
            if hstack is None:
                hstack = original
            else:
                hstack = np.hstack([hstack, original])
        if vstack is None:
            vstack = hstack
        else:
            vstack = np.vstack([vstack, hstack])
    cv2.imwrite(name+".png",vstack)

    return

from utils import confusion_matrix
def clusterResults(extractor, desc,dist, input_labels):

    if extractor =="SIFT":
        #Predict using K-means
        kmean = KMeansCluster(n_clusters = 7)
        kmean.fit(desc)
        k_means_labels  = sortLabels(input_labels,kmean.predict(desc))
        kmean.performance(input_labels,k_means_labels)
        confusion_matrix(input_labels,k_means_labels, save_name = "confusion_matrix_kmean.png")
        #Predict using SpectralClustering
        spectral = SpectralCluster(n_clusters = 7)
        spectral_labels  = sortLabels(input_labels,spectral.predict(desc))
        spectral.performance(input_labels,spectral_labels)

        #predict using HierarchicalClustering
        hierarchical = ClusterAlgorithm()
        hierarchical_labels = sortLabels(input_labels,HierarchicalClustering().draw_dendogram(dist))
        hierarchical.performance(input_labels,hierarchical_labels)

    elif extractor =="SURF":
        #Predict using K-means
        kmean = KMeansCluster()
        kmean.fit(desc)
        k_means_labels  = sortLabels(input_labels,kmean.predict(desc))
        kmean.performance(input_labels,k_means_labels)

        #Predict using SpectralClustering
        spectral = SpectralCluster()
        spectral_labels  = sortLabels(input_labels,spectral.predict(desc))
        spectral.performance(input_labels,spectral_labels)

        #predict using HierarchicalClustering
        hierarchical = ClusterAlgorithm()
        hierarchical_labels = sortLabels(input_labels,HierarchicalClustering().draw_dendogram(dist))
        hierarchical.performance(input_labels,hierarchical_labels)

    elif extractor =="HOG":
        #Predict using K-means
        kmean = KMeansCluster()
        kmean.fit(desc)
        k_means_labels  = sortLabels(input_labels,kmean.predict(desc))
        kmean.performance(input_labels,k_means_labels)

        #Predict using SpectralClustering
        spectral = SpectralCluster()
        spectral_labels  = sortLabels(input_labels,spectral.predict(desc))
        spectral.performance(input_labels,spectral_labels)

        #predict using HierarchicalClustering
        hierarchical = ClusterAlgorithm()
        hierarchical_labels = sortLabels(input_labels,HierarchicalClustering().draw_dendogram(desc))
        hierarchical.performance(input_labels,hierarchical_labels)




    return

def sortLabels(y_true,y_pred):
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(-w)

    new_pred = np.zeros(len(y_pred), dtype=np.int64)
    for i in range(len(y_pred)):
        new_pred[i] = ind[1][y_pred[i]]

    return new_pred
