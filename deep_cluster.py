#################
#Network class builds and train a network using the networks defined in the AutoEncoder class
#################
#https://amitness.com/2020/04/deepcluster/




import tensorflow as tf
import numpy as np
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
import os
import pickle


from cbof import BoF_Pooling, initialize_bof_layers
from preprocessing import Preprocessing, PreprocessingFromDataframe

import clustering
from clustering import preprocess_features

from clustering_algorithms import * #TSNEAlgo, PCAAlgo, KMeansCluster, SpectralCluster, ClusterAlgorithm


from utils import accuracy, confusion_matrix

from deep_neural_networks import VGG_BATCHNORM, RESNET101,RESNET50, COAPNET, RESNET, BOF_MODELS





from load_dataset import importWHOI, importKaggleTrain, importKaggleTest
class DeepCluster:
    data_x = None
    data_y = None
    data_y_last = None
    epochs = 2

    NUM_CLUSTER = 200
    input_shape = (224,224,3)
    verbose = 1

    def __init__(self):
        #self.data_x = importWHOI()
        self.data_x, self.data_y = importKaggleTrain()
        #discard labels
        self.data_y = tf.keras.utils.to_categorical(self.data_y, num_classes=121, dtype='float32')
        #self.data_x = x
        #self.data_y = y

    def train(self,arch = 1):
        network_archs = ["coapnet","vgg","resnet"]
        network = network_archs[arch]

        initialize_previous = False

        if network == "vgg":
            if initialize_previous == True:
                model = VGG_BATCHNORM(input_shape=self.input_shape,output_shape = self.NUM_CLUSTER)
                model = loadWeights(model)
            else:
                model = VGG_BATCHNORM(input_shape=self.input_shape,output_shape = self.NUM_CLUSTER)

        if network == "coapnet":
            if initialize_previous == True:
                model = COAPNET(input_shape=self.input_shape,output_shape = self.NUM_CLUSTER)
                model = loadWeights(model)
            else:
                model = COAPNET(input_shape=self.input_shape,output_shape = self.NUM_CLUSTER)

        if network == "resnet":
            if initialize_previous == True:

                model = RESNET101(input_shape=self.input_shape,output_shape = self.NUM_CLUSTER)
                model = loadWeights(model)
            else:
                model = RESNET101(input_shape=self.input_shape,output_shape = self.NUM_CLUSTER)

        bof = False
        if bof:
            model = BOF_MODELS(backbone = "vgg")

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.09, nesterov=False) #best 0.005
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

        print(model.summary())



        # creating cluster assignments log
        cluster_log = Logger(os.path.join('', 'clusters'))


        ##### Load training data #####
        train_data, _ = importKaggleTrain(depth=1)

        #### if bof -- initialize bof layer ####


        if bof:
            import cv2
            data = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (train_data)])/255
            #data = data[0:5000]
            data=np.stack([data]*3, axis=-1)
            data = data.reshape(-1, 64,64,3)
            #data = data.reshape(100,64,64)
            print(data.shape)
            initialize_bof_layers(model,data)

        ##### Preprocessing #####

        #make training labels
        labels = np.zeros(30336)
        self.data_y_last= labels.copy()

        preprocess_training = PreprocessingFromDataframe(train_data,labels,dataset='Kaggle', num_classes = self.NUM_CLUSTER)
        true_labels = preprocess_training.returnLabels()

        test = False
        if test:
            train_data, _ = importKaggleTest(depth=1)
            import cv2
            data = np.array([cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_LINEAR) for img in (train_data[0:32])])/255
            #data = data[0:32]
            data=np.stack([data]*3, axis=-1)
            data = data.reshape(-1, 224,224,3)

            from utils import visualize_class_activation_map

            visualize_class_activation_map(model, data)

        if network == "coapnet":
            self.epochs = 30
        else:
            self.epochs = 20

        if initialize_previous == False:
            #Preprocess data
            #preprocess_training = Preprocessing(train_data, labels, dataset='Kaggle', num_classes = 100)

            #Return data set without augmentation for prediction
            #predict_dataset = preprocess_training.createPreprocessedDataset()
            #predict_dataset = preprocess_training.returnDataset()

            #### Initialize k means cluster #####
            deepcluster = clustering.__dict__["Kmeans"](self.NUM_CLUSTER)

            ###### Start training ######
            for round in range(self.epochs):
                end = time.time()

                if network == "resnet":
                    # Remove top layer
                    model = tf.keras.Model(inputs=model.input,
                                          outputs=model.layers[-2].output, name = model.name)

                else:
                    # Remove top layer
                    model = tf.keras.Model(inputs=model.input,
                                          outputs=model.layers[-3].output, name = model.name)
                    #print(model.summary())

                predict_dataset = preprocess_training.createPreprocessedDataset()
                # Compute features
                features = compute_features(predict_dataset, model,train_data.shape[0])

                # Get feature information
                if self.verbose == 1:
                    print(features, features.shape,features.dtype)
                    print('features min=%.3f, max=%.3f' % (features.min(), features.max()))

                # Cluster features
                clustering_loss = deepcluster.cluster(features, verbose=1)

                # Get new labels
                train_labels = clustering.cluster_assign(deepcluster.images_lists,labels)

                if self.verbose == 1:
                    print('number of classes = ', len(deepcluster.images_lists))

                    print(train_labels.shape)
                    print(train_labels[0:20])

                    print(self.data_y_last[0:20])

                    from sklearn.metrics.cluster import normalized_mutual_info_score
                    print("NMI old vs new = ", normalized_mutual_info_score(train_labels,self.data_y_last))
                    print("NMI new vs true = ", normalized_mutual_info_score(train_labels,true_labels))


                # Save labels for next comparison round
                self.data_y_last = train_labels.copy()

                # Update labels
                updatecsvfile(train_labels)

                train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=self.NUM_CLUSTER, dtype='float32')
                preprocess_training.updateLabels(train_labels)

                # Make augmented training set
                train_dataset = preprocess_training.createPreprocessedAugmentedDataset()

                #train_dataset = preprocess_training.returnAugmentedDataset()

                # Make model
                if network == "resnet":
                    x = tf.keras.layers.Dense(self.NUM_CLUSTER, activation='softmax')(model.output)
                    model = tf.keras.Model(inputs=model.input,outputs=x,name=model.name)
                else:
                    x = tf.keras.layers.Activation('relu')(model.output)
                    x = tf.keras.layers.Dense(self.NUM_CLUSTER, activation='softmax')(x)
                    model = tf.keras.Model(inputs=model.input,outputs=x,name=model.name)

                model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

                # Train model
                model.fit(train_dataset, steps_per_epoch= 30336 // 32, verbose = 1, epochs = 1)

                end = time.time()

                # Print log
                if self.verbose == 1:
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
            if not bof:
                #Save model
                saveWeights(model)

        ##### Get results #####

        #Load visualization data

        #preprocess_training = PreprocessingFromDataframe()
        #validate_dataset, validate_labels = preprocess_training.createPreprocessedTestDataset()
        #validate_data, _ = importKaggleTest(depth=1)
        #preprocess_validate = Preprocessing(validate_data, validate_labels, dataset='Kaggle', num_classes = 5)
        #validate_dataset = preprocess_validate.returnDataset()
        validate_dataset, validate_labels = preprocess_training.createPreprocessedTestDataset()
        validate_data = preprocess_training.returnImages()


        if network == "resnet":
            # Remove top layer
            model = tf.keras.Model(inputs=model.input,
                                  outputs=model.layers[-2].output)
        else:
            # Remove top layer
            model = tf.keras.Model(inputs=model.input,
                                  outputs=model.layers[-2].output) #Originally 2

        # Extract features
        features = compute_features(validate_dataset, model,validate_data.shape[0])

        from clustering import preprocess_features
        #features = preprocess_features(features)

        if self.verbose == 1:
            print(features, features.shape,features.dtype)

        #Make save directory
        os.makedirs(os.path.join(str("deepcluster_"+network_archs[arch])), exist_ok=True)

        #move into save dir
        os.chdir(str("deepcluster_"+network_archs[arch]))

        #Predict using K-means
        print("KMEANS CLUSTERING")
        kmean = KMeansCluster(n_clusters = 5)
        kmean.fit(features)
        k_means_labels = sortLabels(validate_labels,kmean.predict(features))
        kmean.performance(validate_labels,k_means_labels)

        print("SPECTRAL CLUSTERING")
        #Predict using SpectralClustering
        spectral = SpectralCluster(n_clusters = 5)
        spectral_labels  = sortLabels(validate_labels,spectral.predict(features))
        spectral.performance(validate_labels,spectral_labels)

        print("BIRCH CLUSTERING")
        #Predict using SpectralClustering
        birch = BIRCHCluster(n_clusters = 5)
        birch.fit(features)
        birch_labels  = sortLabels(validate_labels,birch.predict(features))
        birch.performance(validate_labels,birch_labels)

        print("DBSCAN CLUSTERING")
        #Predict using SpectralClustering
        dbscan = DBSCANCluster(n_clusters = 5)
        dbscan_labels  = sortLabels(validate_labels,dbscan.predict(features))
        dbscan.performance(validate_labels,dbscan_labels)

        print("Gaussian CLUSTERING")
        #Predict using SpectralClustering
        gaussian = GaussianMixtureCluster(n_clusters = 5)
        gaussian.fit(features)
        gaussian_labels  = sortLabels(validate_labels,gaussian.predict(features))
        gaussian.performance(validate_labels,gaussian_labels)


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
        PCA.pca_plot(validate_data,spectral_labels,"pca"+str(network)+"spectral","baseline")

        #Move back to prev dir
        os.chdir("../")


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
