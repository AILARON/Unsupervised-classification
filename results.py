##################
#Script for calculating and plotting classification results, t-SNE plots and CAM resuls
#Main function: getNetworkResults
##################


import time
import numpy as np
import os

from auto_encoder import AutoEncoder
from load_dataset import LoadDataset

from utils import visualize_input_output,kernel_inspection,visualize_class_activation_map,visualize_activation_map,confusion_matrix,visualize_class_predictions,accuracy

from clustering_algorithms import HierarchicalClustering, KMeansClustering,TSNEAlgo
from sklearn.cluster import SpectralClustering
#Dataset paths



KAGGLE_ORIGINAL_TRAIN = ""
KAGGLE_AUGMENTED_TRAIN =""
KAGGLE_ORIGINAL_TEST = ""
KAGGLE_AMOMALIES = ""

def tsne_train(encoder, train_x,train_y,save_data_dir,save_name="tsne"):
    """
    Train and prints tsne plots of encoder output
    !!! Deprecated as of newest version of TSNE class
    """

    imgHeight = 64
    imgWidth = 64
    imgNumber = len(train_x)

    encoded_x = encoder.predict(train_x)
    train_y = train_y
    train_x = np.reshape(train_x,(imgNumber,imgHeight,imgWidth))
    print("shapes = ", encoded_x.shape,train_x.shape,train_y.shape)

    TSNE = TSNEAlgo()
    for i in range (1):
        TSNE.tsne_fit(encoded_x,train_x,train_y.astype(int),save_name=save_name,
            save_data_dir = save_data_dir,perplexity = 35)
    return TSNE

def anomaly():
    """
    Testing models ability to find data anomalies
    !!! not working in current version
    """

    #Load anomaly dataset
    anomaly_data = LoadDataset("dataset/kaggle_anomalies/",0)
    anomaly_data, anomaly_label, val, val_label = anomaly_data.load_data()
    for i in range (len(anomaly_label)):
        anomaly_label[i] = anomaly_label[i] + 5

    #Concatinate test and anomaly
    test_anomaly_data = np.vstack((test_data,anomaly_data))
    test_anomaly_label = np.hstack((test_label, anomaly_label))

    """# Get k-means cluster distance
    cluster_model = KMeansClustering()
    cluster_model.train(encoded_train,None)
    cluster_dist = cluster_model.transform(encoded_test_anomaly)

    correct = 0
    wrong = 0
    total = 0
    for i in range(len(cluster_dist)):
        min_distance = np.amin(cluster_dist[i])
        if(min_distance > 4):
            if(test_anomaly_label[i] > 4):
                correct = correct +1
            else:
                wrong = wrong +1

            print("Dist ",min_distance," True label ", test_anomaly_label[i])
    print("Found ",correct," anomalies and ",wrong," wrong")

    decoded = auto.predict(test_anomaly_data)
    errors = []
    # loop over all original images and their corresponding
    # reconstructions
    for (image, recon) in zip(test_anomaly_data, decoded):
    	# compute the mean squared error between the ground-truth image
    	# and the reconstructed image, then add it to our list of errors
    	mse = np.mean((image - recon) ** 2)
    	errors.append(mse)
    # compute the q-th quantile of the errors which serves as our
    # threshold to identify anomalies -- any data point that our model
    # reconstructed with > threshold error will be marked as an outlier
    thresh = np.quantile(errors, 0.4)
    idxs = np.where(np.array(errors) >= thresh)[0]
    print("[INFO] mse threshold: {}".format(thresh))
    print("[INFO] {} outliers found".format(len(idxs)))
    correct = 0
    wrong = 0
    for i in idxs:
        if(test_anomaly_label[i] > 4):
            correct = correct +1
        else:
            wrong = wrong +1
    print("Found ",correct," anomalies and ",wrong," wrong")

    ds = np.zeros(len(test_anomaly_data))
    for i in idxs:
        ds[i] = 1
    tsne(enc, test_anomaly_data,ds,"anomaly_plot","anomaly_plot")"""

def hierarchical(encoder,tsne,true_data,true_labels,save_name ="hierarchical.png"):
    """
    1. Predicts labels using hierarchical clustering
    2. Prints confusion_matrix
    3. Prints t-SNE plot of prediction
    """
    enc_output = encoder.predict(true_data)
    # Hierarchical Clustering
    labels = HierarchicalClustering()
    predictions = labels.draw_dendogram(enc_output,title='Hierarchical Clustering Dendrogram',savetitle = "hierarchical.png")

    # Confusion matrix of hierarchical clustering
    confusion_matrix(true_labels,predictions,save_name = "confusion_matrix_hierarchical.png")

    # Visualize test predictions from hierarchical
    true_data =  np.reshape(true_data,(len(true_data),64,64))
    visualize_class_predictions(true_data,true_labels,predictions)

def kmean(encoder,tsne,true_data,true_label):
    """
    1. Predicts labels using k-means clustering
    2. Prints confusion_matrix
    3. Prints accuracy
    4. Prints t-SNE plot of prediction
    """
    enc_output = encoder.predict(true_data)
    kmean = KMeansClustering()
    kmean.fit(enc_output)
    pred  = kmean.predict(enc_output)
    accuracy(true_label,pred)
    confusion_matrix(true_label,pred, save_name = "confusion_matrix_kmean.png")
    tsne.tsne_plot(true_data,pred,save_data_dir ="kmean",save_name="kmean")

def spectral(encoder,tsne,true_data,true_label):
    """
    1. Predicts labels using spectral clustering
    2. Prints confusion_matrix
    3. Prints accuracy
    4. Prints t-SNE plot of prediction
    """
    enc_output = encoder.predict(true_data)
    model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',
                               assign_labels='kmeans')
    pred = model.fit_predict(enc_output)
    accuracy(true_label,pred)
    confusion_matrix(true_label,pred, save_name = "confusion_matrix_spectral.png")
    tsne.tsne_plot(true_data,pred,save_data_dir ="spectral",save_name="spectral")
    tsne.tsne_plot(true_data,true_label,save_data_dir ="true_label",save_name="true_label")

def predictionTime(encoder,data_):
    """
    Testing model prediction time using k-mean and spectral clustering
    """

    #initialize algorithm
    kmean = KMeansClustering()
    model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',
                               assign_labels='kmeans')


    for i in range(20):
        iterate = 5000*(i+1)
        data = data_[0:iterate,:]
        print(data.shape)
        enc_output = encoder.predict(data)
        kmean.fit(enc_output)
        print("Time predicting "+str(5000*(i+1))+" images")
        print("KMEAN")
        start = time.time()
        enc_output = encoder.predict(data)
        pred  = kmean.predict(enc_output)
        end = time.time()
        print(end - start)


        print("SC")
        start = time.time()
        enc_output = encoder.predict(data)
        pred = model.fit_predict(enc_output)
        end = time.time()
        print(end - start)

def getNetworkResults(model, model_type):
    """
    Main result function
    """

    #Specify wanted results
    plot_input_output = False
    plot_data = False
    plot_class_activation_map = False
    plot_activation_map = False
    plot_kernel_inspection = False
    hierarchical_clustering = False
    kmean_cluster = True
    spectral_cluster  = True
    pred_time = False

    #Load test dataset
    test_data = LoadDataset("dataset/kaggle_original_train/",0)
    test_data, test_label, val, val_label = test_data.load_data()

    # make save directory
    os.makedirs(os.path.join(model_type), exist_ok=True)
    os.chdir(model_type)

    # get autoencoder, encoder and decoder
    AUTO, ENC, DEC = model.getModel()

    #Predict encoder output
    encoded = ENC.predict(test_data)

    #Fit the tsne algorithm
    TSNE = TSNEAlgo()
    TSNE.tsne_fit(encoded,perplexity = 35)

    if plot_input_output == True:
        # Visualize input and output from autoencoder
        visualize_input_output(AUTO,test_data,model_type)

    if model_type =="globalAverage" and plot_class_activation_map == True:
        visualize_class_activation_map(model, test_data)

    if plot_activation_map == True:
        visualize_activation_map(model, test_data)

    if plot_data == True:
        #Predict results using TSNE
        TSNE.tsne_plot(test_data,test_label,model_type,model_type)

    if plot_kernel_inspection == True:
        kernel_inspection(model,test_data)

    if hierarchical_clustering == True:
        hierarchical(ENC,TSNE,test_data,test_label,save_name ="hierarchical.png")

    if kmean_cluster == True:
        kmean(ENC,TSNE,test_data,test_label)

    if spectral_cluster == True:
        spectral(ENC,TSNE,test_data,test_label)

    if pred_time == True:
        os.chdir("..")
        data = LoadDataset("dataset/kaggle_augmented_train_new/",0)
        data, test_label, val, val_label = data.load_data()
        os.chdir(model_type)
        predictionTime(ENC,data)

def getDECNetworkResults(dec,enc):
    #Load test dataset
    test_data = LoadDataset("dataset/kaggle_original_train/",0)
    test_data, test_label, val, val_label = test_data.load_data()

    big_data = LoadDataset("dataset/kaggle_augmented_train_new/",0)
    big_data, _,_, _ = big_data.load_data()

    # make save directory
    os.makedirs(os.path.join("dec"), exist_ok=True)
    os.chdir("dec")

    encoded = enc.predict(test_data)
    q, _ = dec.predict(test_data, verbose=0)
    y_pred = q.argmax(1)

    print(y_pred)
    confusion_matrix(test_label.astype(np.int64),y_pred)

    #Take prediction time
    for i in range(20):
        iterate = 5000*(i+1)
        data = big_data[0:iterate,:]
        print(data.shape)
        print("KMEAN")
        start = time.time()
        q, _ = dec.predict(data, verbose=0)
        y_pred = q.argmax(1)
        end = time.time()
        print(end - start)



    train_x = np.reshape(test_data,(3720,64,64))



    TSNE = TSNEAlgo()
    TSNE.tsne_fit(encoded,perplexity = 35)

    TSNE.tsne_plot(train_x,y_pred.astype(int),save_name="Pred",
        save_data_dir = "dec")
    TSNE.tsne_plot(train_x,test_label.astype(int),save_name="True",
        save_data_dir = "dec")

def getFullyConnectedNetworkResults():
    return
