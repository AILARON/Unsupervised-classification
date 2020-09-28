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
import numpy as np
from descriptors import Descriptor, BFMatcher

from clustering_algorithms import *

import tensorflow as tf
import scipy.spatial.distance as ssd
from load_dataset import importKaggleTest
import cv2

def calculate_distances(input_data, desc ='SIFT'):
    desc_len = 64       # SURF descriptor length
    if desc == 'SIFT':
        desc_len = 128

    X_desc = []
    temp = []
    for image in input_data: #.iloc[:, 0]:
        print(image)
        #image = Image(x).image_read(resize=False)
        img = np.float64(image) / np.max(image)
        #img = image.astype('float32')
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


def traditionalExtraction(exct = 0):
    extractor_types = ["SIFT","SURF","HOG"]
    extractor_method = extractor_types[exct]

    #Import data
    input_data, input_labels = importKaggleTest()
    input_data_center = input_data - input_data.mean()
    #input_data = input_data[:1000,:,:]


    if extractor_method == "SIFT":
        X_I, X_SIFT, img = calculate_distances(input_data, desc='SIFT')

        #Visualize sift key points
        printImages(img,name=extractor_method)

        #Calculate and print cluster results
        clusterResults(extractor_method,X_I, X_SIFT, input_labels)

        labels_sift = \
            HierarchicalClustering() \
                    .draw_dendogram(X_SIFT,
                                title='Hierarchical Clustering Dendrogram of the SIFT Descriptors')


    elif extractor_method == "SURF":
        X_U, X_SURF = calculate_distances(input_data, desc='SURF')

        #Visualize sift key points
        printImages(img,name=extractor_method)

        #Calculate and print cluster results
        clusterResults(extractor_method,X_I, X_SIFT, input_labels)

        labels_sift = \
            HierarchicalClustering() \
                    .draw_dendogram(X_SIFT,
                                title='Hierarchical Clustering Dendrogram of the SIFT Descriptors')

    #Get features from deep neural network for better visualization
    features = extractDeepFeatures(input_data_center)

    #Make TSNE plot using sift descriptors and predicted sift labels
    tsne = TSNEAlgo()
    tsne.tsne_fit(X_I,perplexity=35)
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
    """
    # Hierarchical Clustering
    labels = HierarchicalClustering()\
        .draw_dendogram(X,
                        title='Hierarchical Clustering Dendrogram')
    #labels = MeanShiftAlgo().meanshift_fit(X)
    #PrincipleComponentAnalysis().pca_fit(labels,X)
    TSNEAlgo().tsne_fit(X, input_data, labels)

    labels_hog = \
        HierarchicalClustering()\KMeansCluster
            .draw_dendogram(X_hog,
                            title='Hierarchical Clustering Dendrogram of the HOG Descriptors')
    #labels = MeanShiftAlgo().meanshift_fit(X_hog)
    #PrincipleComponentAnalysis().pca_fit(labels_hog, X_hog)
    TSNEAlgo().tsne_fit(X_hog, input_data, labels_hog, title='HOG Descriptors TSNE Representation')
    TSNEAlgo().tsne_fit(X, input_data, labels_hog, title='HOG Labeled TSNE Representation')
    """
    """labels_sift = \
        HierarchicalClustering() \
                .draw_dendogram(X_SIFT,
                            title='Hierarchical Clustering Dendrogram of the SIFT Descriptors')

    #print(X_SIFT)
    #labels_sift = KMeansCluster().fit(X_I).predict(X_I)



    print(X_I.shape)
    print(input_data.shape)
    print(labels_sift.shape)


    #Print a 4x4 image array
    printImages(img)


    #Get features from deep neural network for better visualization
    features = extractDeepFeatures(input_data_center)

    #Make TSNE plot using sift descriptors and predicted sift labels
    tsne = TSNEAlgo()
    tsne.tsne_fit(X_I,perplexity=35)
    tsne.tsne_plot(input_data, labels_sift, "SIFT","SIFT")

    #Make TSNE plots using features from deep network, predicted sift labels and true labels
    tsne = TSNEAlgo()
    tsne.tsne_fit(features,perplexity=35)
    tsne.tsne_plot(input_data, data_y, "SIFT_COAP_TRUE","SIFT")
    tsne.tsne_plot(input_data, labels_sift,"SIFT_COAP_PRED","SIFT")
    """
    """
    labels_surf = \
        HierarchicalClustering() \
            .draw_dendogram(X_SURF,
                            title='Hierarchical Clustering Dendrogram of the SURF Descriptors')
    TSNEAlgo().tsne_fit(X_U, input_data, labels_surf, title='SURF Descriptors TSNE Representation')
    TSNEAlgo().tsne_fit(X, input_data, labels_surf, title='SURF Labeled TSNE Representation')"""






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
    train_data_3d=np.stack([input_data]*3, axis=-1)
    train_data_3d = train_data_3d.reshape(-1, 64,64,3)

    #Make tensorflow dataset
    visualize_dataset = tf.data.Dataset.from_tensor_slices((train_data_3d))
    visualize_dataset = visualize_dataset.batch(32)

    #Remove softmax layer
    output = model.layers[-2].output
    model_extractor = tf.keras.models.Model(inputs=model.input,
                       outputs=output)
    #Extract features
    features = model_extractor.predict(visualize_dataset)

    return features

def printImages(image_data, name ="name"):
    hstack = None
    vstack = None
    for i in range (4):
        hstack = None
        for j in range (4):
            original = image_data[i*4 +j]
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

def clusterResults(extractor, desc,dist, input_labels):

    if extractor =="SIFT":
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
    from sklearn.utils.linear_assignment_ import linear_assignment

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
        new_pred[i] = ind[y_pred[i]][1]

    return new_pred
