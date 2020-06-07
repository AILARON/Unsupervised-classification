#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the CLUSTERING class
# Author: Aya Saad
# Date created: 24 September 2019
#
#################################################################################################################
from utils import fashion_scatter, color_list, tile_scatter

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from time import time
import pandas as pd

from sklearn.manifold import TSNE

## Hierarchical Clustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist

from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np

# some settings
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation


class ClusterAlgorithm:
    model = None

    def train(self, train_x, train_y):
        # Train the model
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

class KNearestClustering(ClusterAlgorithm):
    n_neighbors = 2

    def build_model(self):
        '''
        The k-nearest neighbor classifier is a “lazy learning” algorithm where nothing is actually “learned”.
        Instead, the k-Nearest Neighbor (k-NN) training phase simply accepts a set of feature vectors and labels and stores them
        Classifying a new feature vector is done as follows:
        - it accepts the feature vector,
        - computes the distance to all stored feature vectors (normally using the Euclidean distance
        - sorts them by distance
        - returns the top k “neighbors” to the input feature vector.
        Each of the k neighbors vote as to what they think the label of the classification is.
        Build the model
        :return: model
        '''
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    pass


class KMeansClustering(ClusterAlgorithm):
    n_clusters = 6
    n_init = 20

    def build_model(self):
        '''
        Build the model
        :return: model
        '''
        self.model = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)

    pass


class MeanShiftAlgo():
    def meanshift_fit(self, X):
        ms = MeanShift()
        ms.fit(X)

        colors = 10 * color_list

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        n_clusters_ = len(np.unique(labels))
        print('Number of estimated clusters from the MeanShift: ', n_clusters_)
        print(type(labels))
        print(labels)

        print(cluster_centers)
        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], c=colors[labels[i]], markersize=10)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                    marker='x', s=150, linewidths=5, zorder=10)
        plt.title('MeanShift Clustering ')
        plt.show()
        return labels

class PrincipleComponentAnalysis():
    def pca_fit(self, labels, X):
        time_start = time()

        x = StandardScaler().fit_transform(X)
        pca = PCA(n_components=10)
        principalComponents = pca.fit_transform(x)
        print('-- PCA DONE! Time elapsed. {} seconds'.format(time() - time_start))
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9',
                                              'pca10'])

        print(pca.explained_variance_ratio_)
        print(sum(pca.explained_variance_ratio_) * 100)
        top_two_comp = principalDf[['pca1', 'pca2']]
        print('top_two_comp.values', top_two_comp.values, 'labels' ,labels)
        f_pca, ax_pca, _, _ = fashion_scatter(top_two_comp.values, labels)
        ax_pca.set_xlabel('Principal Component 1', fontsize=10)
        ax_pca.set_ylabel('Principal Component 2', fontsize=10)
        ax_pca.set_title('2 Component PCA', fontsize=10)
        f_pca.show()

        return


class TSNEAlgo():
    def tsne_fit(X, input_data, labels,save_name ='TSNE',save_data_dir = 'tsne_plot', perplexity = 5,title='TSNE'):
        x = StandardScaler().fit_transform(X)
        #time_start = time()
        RS = 123
        tsne = TSNE(random_state=RS,perplexity =perplexity,
            n_iter = 100000, n_iter_without_progress =500).fit_transform(x)

        ## -- drawing the scatterplot from TSNE
        fashion_scatter(tsne, labels, save_name,save_data_dir)

        ## -- drawing the full images on TSNE
        tile_scatter(tsne, input_data,labels,save_name,save_data_dir)

        return

class HierarchicalClustering():
    '''
    HierarchicalClustering Algorithm
    Deduces the cut-off distance from the dendogram = median distances + 2 * std of distances
    The algorithm suggests the number of clusters based on the cut-off distance
    '''
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

    def draw_dendogram(self, X, title='Hierarchical Clustering Dendrogram'):
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
        max_d = np.median(Z[:, 2]) + 2 * np.std(Z[:, 2])

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
            max_d=max_d,  # plot a horizontal cut-off line
            title=title
        )
        plt.show()
        clusters = fcluster(Z, max_d, criterion='distance')
        n_clusters = len(np.unique(clusters))
        print(np.unique(clusters))
        print('Number of estimated clusters from max_d: ', n_clusters)
        clusters = [x - 1 for x in clusters]
        np_clusters = np.asarray(clusters)
        print(type(np_clusters))
        print(np_clusters)





        return np_clusters
