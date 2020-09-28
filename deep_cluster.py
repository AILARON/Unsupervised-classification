#################
#Network class builds and train a network using the networks defined in the AutoEncoder class
#################
#https://amitness.com/2020/04/deepcluster/
import tensorflow as tf
from sklearn.cluster import KMeans

import tensorflow as tf
import numpy as np
from cbof import BoF_Pooling
from regularizer import SparseActivityRegularizer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


from clustering_algorithms import TSNEAlgo, KMeansCluster
from utils import accuracy, confusion_matrix
from sklearn.model_selection import train_test_split
from load_dataset import importKaggleTrain, importKaggleTest
#import numpy as np
#import cv2
#import tensorflow as tf
#import tensorflow.keras.backend as K
#import matplotlib.pyplot as plt
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from deep_neural_networks import VGG_BATCHNORM, RESNET101, COAPNET, RESNET
import os
import pickle


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

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

def assignSameLabel(old,new):
    from sklearn.utils.linear_assignment_ import linear_assignment

    D = max(old.max(), new.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(new.size):
        w[new[i], old[i]] += 1
    ind = linear_assignment(-w)
    #print(ind)
    newnew = np.zeros(len(new), dtype=np.int64)
    for i in range(len(new)):
        newnew[i] = ind[new[i]][1]
        #for val in ind:
        #    if val[0] == new[i]:
        #        newnew[i] == val[1]

    print("newnew",newnew)
    #for i in range(len(old)):
        #print(j[1] for j in ind)
        #newnew[i] = ind[new[i],]


    #acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    #print("ACC = ",acc)
    return newnew




def kmean_two(features,true_label):
    """
    1. Predicts labels using k-means clustering
    2. Prints confusion_matrix
    3. Prints accuracy
    4. Prints t-SNE plot of prediction
    """
    #enc_output = encoder.predict(true_data)
    kmean = KMeansClustering()
    kmean.fit(features)
    pred  = kmean.predict(features)
    accuracy(true_label,pred)
    #confusion_matrix(true_label,pred, save_name = "confusion_matrix_kmean.png")
    #tsne.tsne_plot(true_data,pred,save_data_dir ="kmean",save_name="kmean")


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


def compute_features(dataset, model,N):

    print('Compute features')
    #batch_time = AverageMeter()
    end = time.time()
    #model.eval()
    # discard the label information in the dataloader
    #data_batch = np.zeros((32, 224,224), dtype='float32')
    for i, (input_tensor, _) in enumerate(dataset):
        #input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        if i <= (N//32 - 1):
            aux = model(input_tensor) #model(input_var).data.cpu().numpy()
        else:
            print("yo")
            input_tensor = input_tensor[0:N-i* 32]
            aux = model(input_tensor)
            print(aux)
        #print(aux)
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        #aux = aux #.astype('float32')
        if i <= (N//32 - 1):
            #print(N//32 - 1)
            #print(i)
            #print(aux.shape)
            features[i * 32: (i + 1) * 32] = aux
        else:
            #print(i+1)
            # special treatment for final batch
            #print(aux)
            features[i * 32:] = aux


        #print(i)

        if not i <= N//32 - 1:
            break
        # measure elapsed time
        #batch_time.update(time.time() - end)
        #end = time.time()



    return features


def updatecsvfile(labels):

    import csv
    r = csv.reader(open('output.csv')) # Here your csv file
    lines = list(r)

    for i in range(len(lines)-1):
        lines[i+1][1] = [int(labels[i])]

    with open('output.csv', 'w') as file:
        writer = csv.writer(file)

        for d in lines:
            writer.writerow(d)


NUM_CLUSTER = 121

from load_dataset import importWHOI, importKaggleTrain
class DeepCluster:
    data_x = None
    data_y = None
    data_y_last = None
    epochs = 10

    def __init__(self):
        #self.data_x = importWHOI()
        self.data_x, self.data_y = importKaggleTrain()
        #discard labels
        self.data_y = tf.keras.utils.to_categorical(self.data_y, num_classes=NUM_CLUSTER, dtype='float32')
        #self.data_x = x
        #self.data_y = y

    def train(self):

        #import training variables
        x = self.data_x
        y = 0


        #import Model
        #model = self.COAPNET()
        model = self.VGG()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.01, nesterov=False)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print(model.summary())

        #PCA reduction before clustering
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        pca = PCA(n_components=256, whiten = True)

        #Run training algorithm
        for i in range(self.epochs):

            #Label the dataset
            if i%20 ==0:
                #remove top layer
                output = model.layers[-2].output
                model_extractor = Model(inputs=model.input,
                                   outputs=output)

                #Extract features:
                print("Extract features")
                x_new = model_extractor.predict(x)

                #Perform PCA reduction and normalization
                x_new = pca.fit_transform(x_new)
                #x_new = normalize(x_new, norm='l2', axis=0, copy=False)

                #Perform clustering
                print("K-means clustering")
                y = KMeans(n_clusters=NUM_CLUSTER,random_state= np.random.randint(0,1000)).fit_predict(x_new)
                print("Number of classes = ", len(set(y)))

                #Calculate number in same cluster
                if i > 0:
                    accuracy(yold,y)
                    #y = assignSameLabel(yold,y)
                yold = y

                #To categorical
                y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLUSTER, dtype='float32')

                #add top layer
                layer = tf.keras.layers.Dense(NUM_CLUSTER, activation='softmax')(output)
                model = Model(inputs=model.input,
                                   outputs=layer)
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.01, nesterov=False)
                #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

            #get images from dataset
            #images = np.concatenate([img for img, label in x], axis=0)


            #Build new training set using clustering labels
            train_dataset = tf.data.Dataset.from_tensor_slices((x, self.data_y))
            train_dataset = train_dataset.shuffle(100000).batch(32)



            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)


            #datagen.fit(x)
            # fits the model on batches with real-time data augmentation:
            #model.fit(datagen.flow(x, y, batch_size=32),
            #    steps_per_epoch=len(x)//32, epochs=20)


            model.fit(train_dataset, epochs=1, verbose = 1, steps_per_epoch =len(x)//32 )

        #Fit the tsne algorithm

        #remove top layer
        output = model.layers[-2].output
        model_extractor = Model(inputs=model.input,
                           outputs=output)
        #print(model.sumamry())

        #Extract features:
        from load_dataset import LoadDataset

        #Load test dataset
        test_data = LoadDataset("dataset/kaggle_original_train/",0,False)
        test_data, test_label = test_data.load_data()

        print("Extract features")
        x_new = model_extractor.predict(test_data)
        print(x_new.shape)


        #features = pca.fit_transform(x_new)
        features = normalize(x_new, norm='l2', axis=0, copy=False)
        kmean_two(features,test_label)

        TSNE = TSNEAlgo()
        TSNE.tsne_fit(x_new,perplexity = 35)

        TSNE.tsne_plot(test_data,test_label,"test_vgg","test_vgg")

        kmean(model_extractor,TSNE, test_data,test_label)


        return

    def new_train(self,arch = 1):
        network_archs = ["coapnet","vgg","resnet"]
        network = network_archs[arch]

        initialize_previous = False

        if network == "vgg":
            if initialize_previous == True:
                model = VGG_BATCHNORM(input_shape=(224,224,3),output_shape = 200)
                model = loadWeights(model)

            else:
                model = VGG_BATCHNORM(output_shape = 100)

        if network == "coapnet":
            if initialize_previous == True:
                model = COAPNET(input_shape=(224,224,3),output_shape = 200)
                model = loadWeights(model)

            else:
                model = COAPNET(input_shape=(64,64,3),output_shape = 100)

        if network == "resnet":
            if initialize_previous == True:
                #model = RESNET50()
                #model = loadWeights(model)
                from res import ResnetBuilder
                #model= ResnetBuilder.build_resnet_18((64,64,3),121)
                model = RESNET101(output_shape = NUM_CLUSTER)
                model = loadWeights(model)
            else:
                #input = tf.keras.layers.Input([None, None, 3])
                #x = tf.keras.applications.resnet_v2.preprocess_input(input, data_format=None)
                model = RESNET101(input_shape=(64,64,3),output_shape = 100)
                ##output = core(x)
                #model = tf.keras.Model(inputs=input, outputs=output)
                #from res import ResnetBuilder
                #model= ResnetBuilder.build_resnet_34((224,224,3),121)

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.09, nesterov=False) #best 0.005
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

        print(model.summary())

        # creating cluster assignments log
        cluster_log = Logger(os.path.join('', 'clusters'))


        #load training data
        #original_data, original_labels = importKaggleTrain(depth=1)

        #train_data = original_data#[0:400]
        #print(train_data)
        #train_labels = original_labels#[0:400]

        #Preprocessing
        from preprocessing import Preprocessing, PreprocessingFromDataframe
        #labels = np.zeros(train_labels.shape[0])
        #train_labels = tf.keras.utils.to_categorical(labels, num_classes=100, dtype='float32')
        #predict_dataset = Preprocessing().createPreprocessedPredictDataset(train_data, train_labels)

        #preprocess_training = PreprocessingFromDataframe(dataset='Kaggle', num_classes = 100)
        train_data, train_labels = importKaggleTrain(depth=1)
        labels = np.zeros(30336)
        preprocess_training = Preprocessing(train_data, labels, dataset='Kaggle', num_classes = 100)

        predict_dataset = preprocess_training.returnDataset()

        #for element in predict_dataset:
        #    print(element)

        import clustering

        # clustering algorithm to use
        deepcluster = clustering.__dict__["Kmeans"](100)
        #Run training algorithm
        #print("LABEL SIze",train_data.shape[0])

        old_labels = labels
        #predict_dataset = preprocess_training.createPreprocessedDataset()
        for round in range(20):
            end = time.time()

            #remove top layer
            #output = model.layers[-2].output
            model = Model(inputs=model.input,
                                  outputs=model.layers[-3].output)
            #model = tf.keras.models.Sequential(model.layers[:-2])
            #print(model.summary())
            #get dataset
            #predict_dataset = preprocess_training.returnDataset()

            # get the features for the whole dataset
            #features = model_extractor.predict(predict_dataset)

            features = compute_features(predict_dataset, model,30336)
            print(features, features.shape,features.dtype)
            print('features min=%.3f, max=%.3f' % (features.min(), features.max()))

            clustering_loss = deepcluster.cluster(features, verbose=1)

            #print(clustering_loss)
            print(len(deepcluster.images_lists))

            train_labels = clustering.cluster_assign(deepcluster.images_lists,labels)
            print(len(train_labels))
            print(train_labels[0:20])
            print(old_labels[0:20])
            old_labels = train_labels

            #updatecsvfile(train_labels)
            #update labels
            train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=100, dtype='float32')
            preprocess_training.updateLabels(train_labels)
            train_dataset = preprocess_training.returnAugmentedDataset()

            #make training dataset
            #train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=200, dtype='float32')
            #train, test, train_l, test_l = train_test_split(train_data, train_labels, test_size=0.1)

            #train_dataset = preprocess_training.createPreprocessedAugmentedDataset()


            #Make model
            #remove top layer
            #output = model.layers.output

            #output.trainable = False  # Freeze ResNet50Base.

            #assert output.trainable_weights == []  # ResNet50Base has no trainable weights.
            #assert len(model.trainable_weights) == 2  # Just the bias & kernel of the Dense layer.

            x = tf.keras.layers.Activation('relu')(model.output)
            #model.trainable = False
            x = tf.keras.layers.Dense(100, activation='softmax')(x)
            model = tf.keras.models.Model(inputs=model.input,outputs=x,name=network_archs[arch])

            model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
            #print(model.summary())
            #train_dataset = Preprocessing(dataset='Kaggle', num_classes = 200).createPreprocessedAugmentedDataset(train, train_l)
            #test_dataset = Preprocessing(dataset='Kaggle', num_classes = 200).createPreprocessedDataset(test,test_l)
            #train_dataset = preprocess_training.returnAugmentedDataset()
            model.fit(train_dataset, steps_per_epoch= 30336 // 32, verbose = 1, epochs = 1)
            #30336 // 32
            #model.trainable = True
            #model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
            #model.fit(train_dataset, steps_per_epoch= 30336 // 32, verbose = 1, epochs = 1)

            # train network with clusters as pseudo-labels
            end = time.time()

            # print log

            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  #'ConvNet loss: {3:.3f}'
                  .format(round, time.time() - end, clustering_loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')

            # save cluster assignments
            cluster_log.log(deepcluster.images_lists)





        from clustering_algorithms import TSNEAlgo, PCAAlgo

        from clustering_algorithms import KMeansCluster, SpectralCluster

        from clustering import preprocess_features


        #Load visualization data
        #preprocess_training = PreprocessingFromDataframe()
        #validate_dataset, validate_labels = preprocess_training.createPreprocessedTestDataset()
        validate_data, validate_labels = importKaggleTest(depth=1)

        preprocess_validate = Preprocessing(validate_data, validate_labels, dataset='Kaggle', num_classes = 5)

        validate_dataset = preprocess_validate.returnDataset()



        #Remove softmax layer
        #output = model.layers[-2].output
        #model_extractor = tf.keras.models.Model(inputs=model.input,
        #                   outputs=output)

        #print("IS THIS ERROR?")
        #Extract features
        #features = model_extractor.predict(validate_dataset, verbose = 1,steps=30336//32)
        #print(features.shape)
        model = Model(inputs=model.input,
                              outputs=model.layers[-2].output) #Originally 2
        #model = tf.keras.models.Sequential(model.layers[:-2])
        print(model.summary())
        #get dataset
        #predict_dataset = preprocess_training.returnDataset()

        # get the features for the whole dataset
        #features = model_extractor.predict(predict_dataset)

        features = compute_features(validate_dataset, model,3720)
        print(features, features.shape,features.dtype)

        #features = preprocess_features(features)
        #visualize_labels = visualize_labels[0:3712,:]
        #Make save directory
        os.makedirs(os.path.join(str("deepcluster_"+network_archs[arch])), exist_ok=True)

        #move into save dir
        os.chdir(str("deepcluster_"+network_archs[arch]))

        #Predict using K-means
        kmean = KMeansCluster(n_clusters = 5)
        kmean.fit(features)
        k_means_labels = sortLabels(validate_labels,kmean.predict(features))
        kmean.performance(validate_labels,k_means_labels)

        #Predict using SpectralClustering
        spectral = SpectralCluster(n_clusters = 5)
        spectral_labels  = sortLabels(validate_labels,spectral.predict(features))
        spectral.performance(validate_labels,spectral_labels)

        #Visualize using TSNE
        TSNE = TSNEAlgo()
        TSNE.tsne_fit(features,perplexity = 35)

        import cv2
        validate_data = np.array([cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for img in (validate_data)])


        TSNE.tsne_plot(validate_data,validate_labels,str(network)+"baseline","baseline")
        TSNE.tsne_plot(validate_data,k_means_labels,str(network)+"kmeans","baseline")
        TSNE.tsne_plot(validate_data,spectral_labels,str(network)+"spectral","baseline")

        #Visualize using PCA
        PCA = PCAAlgo()
        PCA.pca_fit(features)

        PCA.pca_plot(validate_data,validate_labels,"pca"+str(network)+"baseline","baseline")
        PCA.pca_plot(validate_data,k_means_labels,"pca"+str(network)+"kmeans","baseline")
        PCA.pca_plot(validate_data,k_means_labels,"pca"+str(network)+"spectral","baseline")

        #Move back to prev dir
        os.chdir("../")
