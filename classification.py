############


############
import tensorflow as tf


from neural_network_utils import *
if tf.__version__ == '1.15.0':
    from rot_inv_autoencoder import RotationInvariantAutoencoder

from load_dataset import loadFromDataFrame, importKaggle
import cv2
from preprocessing import Preprocessing, PreprocessingFromDataframe

from deep_neural_networks import VGG_BATCHNORM

from clustering_algorithms import *

from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from DEC import DeepEmbeddedClustering

from utils import confusion_matrix

#CSV load files
KAGGLE_TRAIN = 'csvloadfiles/kaggle_original.csv'
KAGGLE_TEST = 'csvloadfiles/kaggle_five_classes.csv'
KAGGLE_MISSING = 'csvloadfiles/kaggle_missing.csv'
KAGGLE_MISSING_TEST = 'csvloadfiles/kaggle_missing_five.csv'
KAGGLE_TEST_HUNDRED = 'csvloadfiles/kaggle_original_test.csv'


class Classification():
    def __init__(self):

            return

    def getEncoderModel(self,name):
        _,encoder,_ = RotationInvariantAutoencoder().autoencoder_architecture()
        encoder = loadWeights(encoder,name)
        return encoder

    def getDeepClusterModel(self,name):
        model = VGG_BATCHNORM(input_shape=(64,64,3),output_shape = 200)
        return loadWeights(model,name)


    def testClusteringModels(self):
        return

class DEC(Classification):
    def __init__(self):

            return

    def train(self):
        return

    def test(self):
        return

class DAC(Classification):
    def __init__(self):

            return

    def train(self):
        return

    def test(self):
        return

class ClusterModels(Classification):
    def __init__(self):
        return

    def clustering(self,features,labels, n_clusters = 5):

        #Predict using K-means
        print("KMEANS CLUSTERING")
        kmean = KMeansCluster(n_clusters = n_clusters)
        kmean.fit(features)
        k_means_labels = kmean.predict(features)
        accuracy, precision, recall, f1score = kmean.performance(labels,kmean.sortLabels(labels,k_means_labels))
        print("NMI ", normalized_mutual_info_score(labels,k_means_labels))

        #Predict using Spectral Clustering
        print("Spectral CLUSTERING")
        spectral = SpectralCluster(n_clusters = n_clusters)
        spectral.fit(features)
        spectral_labels = spectral.predict(features)
        accuracy, precision, recall, f1score = spectral.performance(labels,spectral.sortLabels(labels,spectral_labels))
        print("NMI ", normalized_mutual_info_score(labels,spectral_labels))


        print("BIRCH CLUSTERING")
        #Predict using SpectralClustering
        birch = BIRCHCluster(n_clusters = n_clusters)
        birch.fit(features)
        birch_labels  = birch.predict(features)
        birch.performance(labels, birch.sortLabels(labels,birch.predict(features)))
        print("NMI ", normalized_mutual_info_score(labels,birch_labels))

        print("DBSCAN CLUSTERING")
        #Predict using SpectralClustering
        dbscan = DBSCANCluster(n_clusters = 5)
        dbscan_labels  = dbscan.predict(features)
        dbscan.performance(labels,dbscan.sortLabels(labels,dbscan_labels))
        print("NMI ", normalized_mutual_info_score(labels,dbscan_labels))

        print("Gaussian CLUSTERING")
        #Predict using SpectralClustering
        gaussian = GaussianMixtureCluster(n_clusters = n_clusters)
        gaussian.fit(features)
        gaussian_labels  = gaussian.predict(features)
        gaussian.performance(labels,gaussian.sortLabels(labels,gaussian_labels))
        print("NMI ", normalized_mutual_info_score(labels,gaussian_labels))


    def testAuto(self):
        # Load images
        test_images,test_labels = loadFromDataFrame(KAGGLE_TEST)
        images_black_and_white = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (test_images)])
        test_set = Preprocessing(test_images,test_labels,autoencoder = True).returnImages()

        #for i in range(5):
        i = 0
        #load model
        os.chdir("Auto-Full")
        model = self.getEncoderModel(str(i))
        os.chdir("../")

        # Compute features
        features = model.predict(test_set)

        # Cluster
        #self.clustering(features,test_labels, n_clusters=5)




        TSNE = TSNEAlgo()
        TSNE.tsne_fit(features,perplexity = 35)
        #Predict using K-means
        print("KMEANS CLUSTERING")
        kmean = KMeansCluster(n_clusters = 5)
        kmean.fit(features)
        k_means_labels = kmean.predict(features)
        accuracy, precision, recall, f1score = kmean.performance(test_labels,kmean.sortLabels(test_labels,k_means_labels))

        confusion_matrix(test_labels,kmean.sortLabels(test_labels,k_means_labels),save_name = "k-means_confusion_matrix.png")


        #Predict using Spectral Clustering
        print("Spectral CLUSTERING")
        spectral = SpectralCluster(n_clusters = 5)
        spectral.fit(features)
        spectral_labels = spectral.predict(features)
        accuracy, precision, recall, f1score = spectral.performance(test_labels,spectral.sortLabels(test_labels,spectral_labels))

        confusion_matrix(test_labels,spectral.sortLabels(test_labels,spectral_labels),save_name = "spectral_confusion_matrix.png")

        '''
        print("BIRCH CLUSTERING")
        #Predict using SpectralClustering
        birch = BIRCHCluster(n_clusters = 5)
        birch.fit(features)
        birch_labels  = birch.predict(features)


        print("Gaussian CLUSTERING")
        #Predict using SpectralClustering
        gaussian = GaussianMixtureCluster(n_clusters = 5)
        gaussian.fit(features)
        gaussian_labels  = gaussian.predict(features)
        '''
        TSNE.tsne_plot(images_black_and_white,test_labels,"baseline","baseline")
        TSNE.tsne_plot(images_black_and_white,k_means_labels,"kmeans","baseline")
        TSNE.tsne_plot(images_black_and_white,spectral_labels,"Spectral","baseline")
        #TSNE.tsne_plot(images_black_and_white,birch_labels,"BIRCH","baseline")
        #TSNE.tsne_plot(images_black_and_white,gaussian_labels,"Gaussian","baseline")


        return

    def testDC(self):
        # Load data
        data, _ = importKaggle(train = True)

        preprocess = PreprocessingFromDataframe(data,dataset='Kaggle',
         num_classes = 200,input_shape =(64,64,3))

        validate_dataset = preprocess.createPreprocessedDataset(filename = KAGGLE_TEST)
        validate_labels = preprocess.returnLabels(filename = KAGGLE_TEST)
        validate_data = preprocess.returnImages(filename = KAGGLE_TEST)

        #for i in range(5):
        #load model

        i = 0

        os.chdir("VGG-Full")
        model = self.getDeepClusterModel(str(i))
        os.chdir("../")

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.09, nesterov=False) #best 0.005

        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

        #print(model.summary())

        # Remove top layer
        model = tf.keras.Model(inputs=model.input,
                              outputs=model.layers[-2].output, name = model.name)

        print(validate_data.shape[0])
        # Compute features
        features = compute_features(validate_dataset.repeat(), model,validate_data.shape[0])

        print(features.shape)

        # PCA reduce features
        kmeansfeatures = StandardScaler().fit_transform(features)
        kmeansfeatures = PCA(n_components=512,copy=True, svd_solver='auto',iterated_power='auto').fit_transform(features)

        #print(features.shape)
        # Cluster
        #self.clustering(features,validate_labels, n_clusters = 121)

        #TIME
        predictionTime(model,data)

        import cv2
        validate_data = np.array([cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR) for img in (validate_data)])

        TSNE = TSNEAlgo()
        TSNE.tsne_fit(features,perplexity = 35)
        #Predict using K-means
        print("KMEANS CLUSTERING")
        kmean = KMeansCluster(n_clusters = 5)
        kmean.fit(kmeansfeatures)
        k_means_labels = kmean.predict(kmeansfeatures)
        accuracy, precision, recall, f1score = kmean.performance(validate_labels,kmean.sortLabels(validate_labels,k_means_labels))

        confusion_matrix(validate_labels,kmean.sortLabels(validate_labels,k_means_labels),save_name = "k-means_confusion_matrix.png")

        #Predict using Spectral Clustering
        print("Spectral CLUSTERING")
        spectral = SpectralCluster(n_clusters = 5)
        spectral.fit(features)
        spectral_labels = spectral.predict(features)
        accuracy, precision, recall, f1score = kmean.performance(validate_labels,spectral.sortLabels(validate_labels,spectral_labels))

        confusion_matrix(validate_labels,kmean.sortLabels(validate_labels,spectral_labels),save_name = "spectral_confusion_matrix.png")

        '''
        print("BIRCH CLUSTERING")
        #Predict using SpectralClustering
        birch = BIRCHCluster(n_clusters = 5)
        birch.fit(features)
        birch_labels  = birch.predict(features)


        print("Gaussian CLUSTERING")
        #Predict using SpectralClustering
        gaussian = GaussianMixtureCluster(n_clusters = 5)
        gaussian.fit(features)
        gaussian_labels  = gaussian.predict(features)

        '''
        TSNE.tsne_plot(validate_data,validate_labels,"baseline","baseline")
        TSNE.tsne_plot(validate_data,k_means_labels,"kmeans","baseline")
        TSNE.tsne_plot(validate_data,spectral_labels,"spectral","baseline")
        #TSNE.tsne_plot(validate_data,birch_labels,"Birch","baseline")
        #TSNE.tsne_plot(validate_data,gaussian_labels,"Gaussian","baseline")


        # Test DEC
        # dec(model, dataset, labels)

        # Test DAC
        # dac(mocel,dataset, labels)

        return


    def dec(self,model = 'DC',train = False):

        if model == 'DC':
            from DEC import DeepEmbeddedClustering
            print('dec')
            # Load data
            data, _ = importKaggle(train = True)

            preprocess = PreprocessingFromDataframe(data,dataset='Kaggle',
             num_classes = 200,input_shape =(64,64,3))

            validate_dataset = preprocess.createPreprocessedDataset(filename = KAGGLE_TEST)
            validate_labels = preprocess.returnLabels(filename = KAGGLE_TEST)
            validate_data = preprocess.returnImages(filename = KAGGLE_TEST)
            predgen, traingen = preprocess.returnGenerator()

            validate_data=np.stack([validate_data]*3, axis=-1)
            validate_data = validate_data.reshape(-1, 64,64,3)

            if train == True:
                for i in range (5):
                    #load model
                    os.chdir("VGG-Full")
                    model = self.getDeepClusterModel(str(i))
                    os.chdir("../")

                    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.09, nesterov=False) #best 0.005

                    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])



                    # Remove top layer
                    model = tf.keras.Model(inputs=model.input,
                                          outputs=model.layers[-2].output, name = model.name)

                    print(model.summary())


                    dec = DeepEmbeddedClustering(model,validate_data,validate_labels,5,predgen, traingen)
                    dec.buildModel()

                    print('init')
                    dec.initializeClusteringLayer()
                    print('BUILD')
                    dec.trainModel()
                    dec.saveModel(str(i))

            if train == False:
                for i in range(5):
                    #load model
                    os.chdir("VGG-Full")
                    model = self.getDeepClusterModel(str(i))
                    os.chdir("../")
                    # Remove top layer
                    model = tf.keras.Model(inputs=model.input,
                                          outputs=model.layers[-2].output, name = model.name)
                    dec = DeepEmbeddedClustering(model,validate_data,validate_labels,5,predgen, traingen)
                    dec.buildModel()
                    dec.loadModel(str(i))

                    # Compute features
                    predictions = compute_features(validate_dataset,dec.getModel() ,validate_data.shape[0])

                    predictions = predictions.argmax(1)
                    perf = ClusterAlgorithm()
                    accuracy, precision, recall, f1score = perf.performance(validate_labels,perf.sortLabels(validate_labels,predictions))
                    print("NMI ", normalized_mutual_info_score(validate_labels,predictions))


                return dec



        if model == 'auto':
            from DEC_old import DeepEmbeddedClustering
            print('auto dec')
            # Load images
            test_images,test_labels = loadFromDataFrame(KAGGLE_TEST)
            images_black_and_white = np.array([cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR) for img in (test_images)])
            pre_train = Preprocessing(test_images,test_labels,autoencoder = False)
            data,labels = pre_train.returnDataset()




            if train == True:

                for i in range (5):
                    #load model
                    os.chdir("Auto-Full")
                    model = self.getEncoderModel(str(i))
                    os.chdir("../")


                    print(model.summary())


                    dec = DeepEmbeddedClustering(model,data,labels,5, pre_train)
                    dec.buildModel()
                    dec.initializeClusteringLayer()
                    print('BUILD')

                    dec.trainModel()
                    dec.saveModel(str(i))
                    dec.loadModel(str(i))
            if train == False:
                for i in range (5):
                    #load model
                    os.chdir("Auto-Full")
                    model = self.getEncoderModel(str(i))
                    os.chdir("../")

                    dec = DeepEmbeddedClustering(model,data,labels,5, pre_train)
                    dec.buildModel()
                    dec.loadModel(str(i))
                    # Compute features
                    predictions = dec.getModel().predict(data)

                    predictions = predictions.argmax(1)
                    perf = ClusterAlgorithm()
                    accuracy, precision, recall, f1score = perf.performance(labels,perf.sortLabels(labels,predictions))
                    print("NMI ", normalized_mutual_info_score(labels,predictions))



        return

def predictionTime(model,input_data):
    """
    Testing model prediction time using k-mean and spectral clustering
    """

    #initialize algorithm
    kmean = KMeansCluster(n_clusters = 5)
    spectral = SpectralCluster(n_clusters = 5)

    input_labels = np.zeros(30336)

    preprocess = Preprocessing(input_data,input_labels,dataset='Kaggle',
     num_classes = 200,input_shape =(64,64,3))
    input_data = preprocess.returnImages()

    for i in range(20):

        iterate = 5000*(i+1)
        data = input_data[0:iterate,:,:]
        labels = input_labels[0:iterate]
        print(data.shape, labels.shape)
        preprocess = Preprocessing(data,labels,dataset='Kaggle',
         num_classes = 200,input_shape =(64,64,3))
        data = preprocess.returnTrainDataset()


        features = compute_features(data, model,5000*(i+1))
        kmean.fit(features)

        print("Time predicting "+str(5000*(i+1))+" images")
        print("KMEAN")
        for i in range(5):
            start = time.time()
            features = compute_features(data, model,5000*(i+1))
            pred  = kmean.predict(features)
            end = time.time()
            print(end - start)


        print("SC")
        for i in range(5):
            start = time.time()
            features = compute_features(data, model,5000*(i+1))
            pred = model.fit_predict(features)
            end = time.time()
            print(end - start)
