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
    encoder =None
    auto_encoder = None


    def __init__(self,auto_encoder,encoder, data,labels, n_clusters):
        """
        Initialize DEC algorithm
        Input: 1. auto_encoder is a prebuilt model
        Input: 2. encoder is a prebuilt model
        Input: 3. data
        Input: 4. labels
        Input: 5. n_clusters define how many classes in the data set
        """
        self.encoder = encoder
        self.auto_encoder = auto_encoder
        self.data = data
        self.n_clusters = n_clusters
        self.labels = labels

        return

    def buildModel(self):
        """
        BuildModel merges AutoEncoder model with the clustering layer and compiles network
        """
        #Make model:
        #pooling = BoF_Pooling(128,spatial_level=1, name="BOF")(enc.get_layer("conv2d_8").output)
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.DEC = Model(inputs=self.encoder.input,
                           outputs=[clustering_layer, self.auto_encoder.output])

        #initialize clustering layer
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(self.data))
        self.DEC.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        self.y_pred_last = np.copy(self.y_pred)

        #compile model
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.01, nesterov=False)
        #pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.DEC.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)

        return

    def trainModel(self):
        """
        TrainModel trains the network
        """

        loss = 0
        index = 0
        maxiter = 20000
        update_interval = 140
        index_array = np.arange(self.data.shape[0])
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
                q, _  = self.DEC.predict(self.data, verbose=0)
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
            idx = index_array[index * batch_size: min((index+1) * batch_size, self.data.shape[0])]
            loss = self.DEC.train_on_batch(x=self.data[idx], y=[p[idx], self.data[idx]])
            index = index + 1 if (index + 1) * batch_size <= self.data.shape[0] else 0


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
