





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

from deep_neural_networks import VGG_BATCHNORM, RESNET101, COAPNET, RESNET

def neuralNetwork(arch):
    network_archs = ["coapnet","vgg","resnet"]
    network = network_archs[arch]


    initialize_previous = False

    if network == "vgg":
        if initialize_previous == True:
            model = VGG_BATCHNORM()
            model = loadWeights(model)

        else:
            model = VGG_BATCHNORM(input_shape=(110,110,3),output_shape = 121)

    if network == "coapnet":
        if initialize_previous == True:
            model = COAPNET()
            model = loadWeights(model)

        else:
            model = COAPNET()

    if network == "resnet":
        if initialize_previous == True:
            #model = RESNET50()
            #model = loadWeights(model)
            from res import ResnetBuilder
            #model= ResnetBuilder.build_resnet_18((64,64,3),121)
            model = RESNET101(output_shape = 7)
            model = loadWeights(model)
        else:
            #input = tf.keras.layers.Input([None, None, 3])
            #x = tf.keras.applications.resnet_v2.preprocess_input(input, data_format=None)
            model = RESNET101(output_shape = 121)
            ##output = core(x)
            #model = tf.keras.Model(inputs=input, outputs=output)
            #from res import ResnetBuilder
            #model= ResnetBuilder.build_resnet_34((224,224,3),121)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])


    print(model.summary())

    #load training data
    train_data, train_labels = importKaggle()
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=121, dtype='float32')

    print(train_labels.shape)
    print(train_labels)


    #Load visualization data
    visualize_data, visualize_labels = importKaggle(train = False)
    #visualize_labels = tf.keras.utils.to_categorical(visualize_labels, num_classes=121, dtype='float32')

    #visualize_data = visualize_data[0:100]
    #visualize_labels = visualize_labels[0:100]

    print(visualize_data.shape)
    print(visualize_labels.shape)

    #print('train_data min=%.3f, max=%.3f' % (train_data.min(), train_data.max()))
    #print('visualize_data min=%.3f, max=%.3f' % (visualize_data.min(), visualize_data.max()))

    #from utils import visualize_input
    #visualize_input(train_data[0:16])

    #Make data between 0-1 ##NOTICED THAT THIS SPEEDS UP TRAINING##
    #train_data = train_data/255
    #visualize_data = visualize_data/255

    #Make data mean centered
    #train_data_centered = train_data - train_data.mean()
    #visualize_data_centered = visualize_data - visualize_data.mean()

    #print('train_data min=%.3f, max=%.3f' % (train_data.min(), train_data.max()))
    #print('visualize_data min=%.3f, max=%.3f' % (visualize_data.min(), visualize_data.max()))



    #Make dataset 3 dimensional
    #train_data_3d=np.stack([train_data_centered]*3, axis=-1)
    #train_data_3d = train_data_3d.reshape(-1, 64,64,3)

    #visualize_data_3d=np.stack([visualize_data_centered]*3, axis=-1)
    #visualize_data_3d = visualize_data_3d.reshape(-1, 224,224,3)

    #divide training data into train and test
    train_data, test_data, train_label, test_label = train_test_split(train_data, train_labels, test_size=0.1)

    #test(model, train_data, train_label, test_data, test_label)

    #Make tensorflow datasets
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    #train_dataset = train_dataset.shuffle(100000).batch(32).repeat()

    #test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    #test_dataset = test_dataset.shuffle(100000).batch(32).repeat()

    #visualize_dataset = tf.data.Dataset.from_tensor_slices((visualize_data_3d, visualize_labels))
    #visualize_dataset = visualize_dataset.batch(64)

    #preprocessing




    #train_data = np.array([tf.image.resize(img, [224,224,3])/255 for img in (train_data)])
    from preprocessing import Preprocessing


    train_dataset = Preprocessing(train_data, train_label,image_width = 110, image_height = 110).returnAugmentedDataset()
    test_dataset = Preprocessing(test_data, test_label,image_width = 110, image_height = 110).returnDataset()
    visualize_dataset = Preprocessing(visualize_data, visualize_labels,image_width = 110, image_height = 110).returnDataset()


    #Early stopping criterion
    stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=30, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False)

    if initialize_previous == False:
        #Train model
        #model.fit(train_dataset,validation_data =test_dataset,steps_per_epoch=int(math.ceil(1. * train_data.shape[0] / 32)),
        #validation_steps=int(math.ceil(1. * test_data.shape[0] / 32)),verbose = 1, callbacks =[stopping],epochs = 200)
        history = model.fit(train_dataset,validation_data =test_dataset, steps_per_epoch= train_data.shape[0] // 32,
        validation_steps= test_data.shape[0] // 32, verbose = 1, callbacks =[stopping], epochs = 30)
        #Save model
        saveWeights(model)


    #Remove softmax layer
    output = model.layers[-2].output
    model_extractor = tf.keras.models.Model(inputs=model.input,
                       outputs=output)

    print("IS THIS ERROR?")
    #Extract features
    features = model_extractor.predict(visualize_dataset, verbose = 1)
    print(features.shape)

    #visualize_labels = visualize_labels[0:3712,:]
    #Make save directory
    os.makedirs(os.path.join(str("baseline_")+network), exist_ok=True)

    #move into save dir
    os.chdir(str("baseline_")+network)

    #Predict using K-means
    kmean = KMeansCluster(n_clusters = 5)
    kmean.fit(features)
    k_means_labels  = sortLabels(visualize_labels,kmean.predict(features))
    kmean.performance(visualize_labels,k_means_labels)

    #Predict using SpectralClustering
    spectral = SpectralCluster(n_clusters = 5)
    spectral_labels  = sortLabels(visualize_labels,spectral.predict(features))
    spectral.performance(visualize_labels,spectral_labels)

    #Visualize using TSNE
    TSNE = TSNEAlgo()
    TSNE.tsne_fit(features,perplexity = 35)

    visualize_data = np.array([cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for img in (visualize_data)])
    print(visualize_labels)
    print(k_means_labels)


    TSNE.tsne_plot(visualize_data,visualize_labels,str(network)+"baseline","baseline")
    TSNE.tsne_plot(visualize_data,k_means_labels,str(network)+"kmeans","baseline")
    TSNE.tsne_plot(visualize_data,spectral_labels,str(network)+"spectral","baseline")

    #Visualize using PCA
    PCA = PCAAlgo()
    PCA.pca_fit(features)

    PCA.pca_plot(visualize_data,visualize_labels,"pca"+str(network)+"baseline","baseline")
    PCA.pca_plot(visualize_data,k_means_labels,"pca"+str(network)+"kmeans","baseline")
    PCA.pca_plot(visualize_data,k_means_labels,"pca"+str(network)+"spectral","baseline")

    #Move back to prev dir
    os.chdir("../")



def sortLabels(y_true,y_pred):
    print(y_pred.shape)
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

    print(new_pred.shape)
    return new_pred

def saveWeights(model):
    # Save JSON config to disk
    json_config = model.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(json_config)
        # Save weights to disk
        print("[Info] saving weights")
        model.save_weights(str(model.name)+"_"+"baseline"+'_weights.h5')

def loadWeights(model):
    print("[Info] loading previous weights")
    try:
        model.load_weights(str(model.name)+"_"+"baseline"+'_weights.h5')
    except:
        print("Could not load weights")

    return model




def test(model, data_x, data_y, test_x, test_y):
    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     #UNDER WAS ADDED LAST
                     horizontal_flip = True,
                     vertical_flip = True,
                     zca_whitening=True)


    data_gen_args_test = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     )

    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    gen = ImageDataGenerator.flow(
        data_x,
        y=data_y,
        batch_size=32,
        shuffle=True,
        sample_weight=None,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        )

    ImageDataGeneratorTest = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args_test)
    gen_test = ImageDataGenerator.flow(
        test_x,
        y=test_y,
        batch_size=32,
        shuffle=False,
        sample_weight=None,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        )


    # Wrap the generator with tf.data
    ds = tf.data.Dataset.from_generator(
        lambda: gen,
        output_types=(tf.float32, tf.float32),
        output_shapes = ([None,64,64,3],[None,121])
    )

    ds_test = tf.data.Dataset.from_generator(
        lambda: gen_test,
        output_types=(tf.float32, tf.float32),
        output_shapes = ([None,64,64,3],[None,121])
    )


    model.fit(ds, steps_per_epoch=data_x.shape[0] // 32, validation_data = ds_test, validation_steps = test_x.shape[0] // 32,
    epochs=100,
    verbose = 1)

    return
