#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the utils
# Author: Aya Saad
# Date created: 24 September 2019
# Changed for this implementation on 3 April 2020 by Eivind Salvesen
#################################################################################################################

## Hierarchical Clustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import os

#KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans

#TSNEAlgo
from sklearn.manifold import TSNE
from utils import fashion_scatter, color_list, tile_scatter
from sklearn.preprocessing import StandardScaler

#SpectralClustering
from sklearn.cluster import SpectralClustering

#PCA
from sklearn.decomposition import PCA

class HierarchicalClustering():
    '''
    HierarchicalClustering Algorithm
    Deduces the cut-off distance from the dendogram = median distances + 2 * std of distances
    The algorithm suggests the number of clusters based on the cut-off distance
    '''

    model = None

    def fancy_dendrogram(self, *args, **kwargs):
        '''
        Apply some fancy visualizations on the dendrogram of the hierarchical clustering
        :param args:
        :param kwargs:
        :return:
        '''
        title = kwargs.pop('title', None)
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title(title)
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    def draw_dendogram(self, X, title='Hierarchical Clustering Dendrogram',savetitle = "hierarchical.png"):
        '''
        Draw the dendogram
        :param X: the dataset
        :return: clusters an array showing each object cluster
        '''
        # generate the linkage matrix
        Z = linkage(X, 'ward')
        #c, coph_dists = cophenet(Z, pdist(X))
        #print('c ', c)

        print('Z[-4:, 2]', Z[-4:, 2])
        print('', Z[:, 2])
        max_d = 20000 #np.median(Z[:, 2]) + 2 * np.std(Z[:, 2])

        # calculate full dendrogram
        plt.figure(figsize=(25, 10))
        plt.title(title)
        plt.xlabel('sample index')
        plt.ylabel('distance')

        self.fancy_dendrogram(
            Z,
            truncate_mode='lastp',
            p=12,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=0.05,  # useful in small plots so annotations don't overlap
            max_d= max_d,  # plot a horizontal cut-off line
            title=title
        )
        plt.savefig(savetitle)
        k = 5
        clusters = fcluster(Z, max_d, criterion='distance') #maxclust
        n_clusters = len(np.unique(clusters))
        print(np.unique(clusters))
        print('Number of estimated clusters from max_d: ', n_clusters)
        clusters = [x - 1 for x in clusters]
        np_clusters = np.asarray(clusters)
        print(type(np_clusters))
        print(np_clusters)

        return np_clusters

    """def fit_predict(self,test_x):
        # generate the linkage matrix
        Z = linkage(X, 'ward')
        clusters = fcluster(Z, max_d, criterion='distance') #maxclust
        return"""

class ClusterAlgorithm:
    model = None

    def train(self, train_x, train_y):
        # Train the model
        print("train")
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        '''
        Predict labels for the given test set
        :param test_x:   Input test set
        :return: predictions
        '''
        print('[INFO] ... Evaluate .... ')
        predictions = []
        for x in test_x:
            pred = self.model.predict(x)
            predictions.append(pred)
        return predictions

    def performance(self, test_y, pred_y):
        '''
        Calculate the performance metrics
        :param test_y:   true y values
        :param pred_y:   predicted y values
        :return: accuracy, precision, recall, f1score
        '''
        # Evaluate the K-Means clustering accuracy.
        accuracy = metrics.accuracy_score(test_y, pred_y)
        precision = metrics.precision_score(test_y, pred_y, average="weighted")
        recall = metrics.recall_score(test_y,pred_y, average="weighted")
        f1score = metrics.f1_score(test_y,pred_y, average="weighted")

        print("Accuracy: {}%".format(100 * accuracy))
        print("Precision: {}%".format(100 * precision))
        print("Recall: {}%".format(100 * recall))
        print("F1 Score: {}%".format(100 * f1score))
        return accuracy, precision, recall, f1score

    def build_model(self):
        pass

    pass

class KMeansCluster(ClusterAlgorithm):
    n_clusters = 5
    n_init = 1000

    def __init__(self,n_clusters = 5, n_init = 100):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.build_model()

    def build_model(self):
        '''
        Build the model
        :return: model
        '''
        self.model = KMeans(max_iter = 10000, n_clusters=self.n_clusters, n_init=self.n_init)

    def fit(self, X):
        return self.model.fit(X)

    def transform(self, X):
        return self.model.transform(X)

    def predict(self, X):
        return self.model.predict(X)
    pass

class SpectralCluster(ClusterAlgorithm):
    n_clusters = 5
    n_init = 1000

    def __init__(self,n_clusters = 5, n_init = 1000):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.build_model()

    def build_model(self):
        '''
        Build the model
        :return: model
        '''
        self.model = SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors',
                                   assign_labels='kmeans')

    def fit(self, X):
        return self.model.fit(X)

    def predict(self, X):
        return self.model.fit_predict(X)
    pass


class TSNEAlgo():
    tsne = None
    def __init__(self):
        return

    def tsne_fit(self, X, perplexity = 5):
        x = StandardScaler().fit_transform(X)
        #time_start = time()
        RS = 123
        self.tsne = TSNE(random_state=RS,perplexity =perplexity,
            n_iter = 100000, n_iter_without_progress =500).fit_transform(x)

        return

    def tsne_plot(self, input_data, labels, save_name = "TSNE", save_data_dir = "tsne_plot"):
        imgHeight = 64
        imgWidth = 64
        imgNumber = len(input_data)
        #input_data = np.reshape(input_data,(imgNumber,imgHeight,imgWidth))


        ## -- drawing the scatterplot from TSNE
        fashion_scatter(self.tsne, labels, save_name,save_data_dir)

        ## -- drawing the full images on TSNE
        tile_scatter(self.tsne, input_data,labels,save_name,save_data_dir)

        return

class PCAAlgo():
    pca = None
    def __init__(self):
        return

    def pca_fit(self, X):
        x = StandardScaler().fit_transform(X)
        #time_start = time()
        RS = 123
        self.pca = PCA(n_components=2,copy=True, svd_solver='auto',
        iterated_power='auto').fit_transform(x)

        return

    def pca_plot(self, input_data, labels, save_name = "pca", save_data_dir = "pca_plot"):
        imgHeight = 64
        imgWidth = 64
        imgNumber = len(input_data)
        #input_data = np.reshape(input_data,(imgNumber,imgHeight,imgWidth))


        ## -- drawing the scatterplot from TSNE
        fashion_scatter(self.pca, labels, save_name,save_data_dir)

        ## -- drawing the full images on TSNE
        tile_scatter(self.pca, input_data,labels,save_name,save_data_dir)

        return
