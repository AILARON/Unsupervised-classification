###############
#Deep Embedded Clustering implementation

#Based on the implementation of DEC found in:
#https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
#https://github.com/Tony607/Keras_Deep_Clustering
#https://github.com/XifengGuo/DEC-keras
###############

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
import metrics
import numpy as np
import tensorflow as tf
from neural_network_utils import *


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(shape =(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddedClustering:
    """
    Class for building and training the Deep Embedded Clustering algorithm
    """
    DEC = None
    n_clusters = None
    y_pred = None
    y_pred_last = None

    data = None
    labels = None
    model =None

    train_generator = None
    predict_generator = None

    predict_dataset =  None
    train_dataset= None


    def __init__(self,model, data,labels, n_clusters,predict_generator,train_generator):
        """
        Initialize DEC algorithm
        Input: 1. model is a prebuilt neural network model
        Input: 2. data
        Input: 3. labels
        Input: 4. n_clusters define how many classes in the data set
        """
        self.model = model
        self.data = data
        self.n_clusters = n_clusters
        self.labels = labels

        self.predict_generator = predict_generator
        self.train_generator = train_generator

        self.predict_dataset = tf.keras.preprocessing.image.NumpyArrayIterator(
        self.data, self.labels, self.predict_generator, batch_size=32)

        self.train_dataset = tf.keras.preprocessing.image.NumpyArrayIterator(
        self.data, self.labels, self.train_generator, batch_size=32)

        return

    def buildModel(self):
        """
        BuildModel merges AutoEncoder model with the clustering layer and compiles network
        """
        #Make model:
        #pooling = BoF_Pooling(128,spatial_level=1, name="BOF")(enc.get_layer("conv2d_8").output)
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.model.output)
        self.DEC = tf.keras.Model(inputs=self.model.input,
                           outputs=clustering_layer)
        #compile model
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.01, nesterov=False)
        #pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.DEC.compile(loss=['kld'], optimizer=optimizer)

        print('Build model finished')
        return

    def initializeClusteringLayer(self):
        #initialize clustering layer
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        self.y_pred = kmeans.fit_predict(compute_features(self.predict_dataset,self.model,3772))
        self.DEC.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        self.y_pred_last = np.copy(self.y_pred)

        #compile model
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.01, nesterov=False)
        #pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.DEC.compile(loss=['kld'], optimizer=optimizer)
        return

    def trainModel(self):
        """
        TrainModel trains the network
        """
        print('train model ')
        loss = 0
        index = 0
        maxiter = 40000
        update_interval = 1
        ##################### CHANGE THIS VALUE TO SHAPE ################# and change the one under!!
        index_array = np.arange(3772)#self.data.shape[0]
        tol = 0.00001 # tolerance threshold to stop training
        batch_size = 32
        y_pred = self.y_pred
        y_pred_last = 0
        y = self.labels

        def target_distribution(q):
            weight = q ** 2 / q.sum(0)
            return (weight.T / weight.sum(1)).T


        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q  = compute_features(self.predict_dataset,self.DEC,3772) #self.DEC.predict(self.data)#, steps = 5)#self.DEC.predict(self.dataset, steps = 118) #compute_features(self.data,self.DEC,3772) #self.DEC.predict(self.data)#, steps = 5)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:

                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc,nmi, ari), ' ; loss=', loss)
                    print(self.DEC.metrics_names)
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break


                self.train_dataset = tf.keras.preprocessing.image.NumpyArrayIterator(
                    self.data, p, self.train_generator, batch_size=32, shuffle = True)


            for i, (data, label) in enumerate(self.train_dataset):
                loss = self.DEC.train_on_batch(x=data, y=label)
                #print(i)
                if i == 118:
                    break
    def saveModel(self):
        saveWeights(self.DEC,'dec')

    def loadModel(self):
        loadWeights(self.DEC,'dec')

    def getModel(self):
        """
        Returns dec model
        """
        return self.DEC, self.encoder


    def evaluateDEC(self):
        # Eval.
        q, _ = self.model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)
