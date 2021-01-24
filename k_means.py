from collections import Counter
from math import log2
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import csv


class K_means_plusplus:
    """
    Note : Cluster names are indicated as numbers (starting from 0)
    """

    def plot_data(self, data):
        """
        Plots the given data in 2D

        inputs:
            data: 2D array filled with points
        """
        x = data[:, 0]
        y = data[:, 1]
        plt.rcParams.update({'figure.figsize': (8, 8), 'figure.dpi': 100})

        plt.scatter(x, y)
        plt.title("Given Data samples")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def load_data(self, file_name):
        """
        Loads and returns the data with the given name.

        inputs:
            file_name: name of the that user wants to load
        """
        col_names = ['X', 'Y']
        data = pd.read_csv(file_name, names=col_names, header=None, usecols=col_names, sep=";")
        return data
    
    def calc_dist(self, X1, X2):
        """
        Calculates the distance between 2 points

        inputs:
            X1, X2: 2 data points
        """
        return (sum((X1 - X2)**2))**0.5
    
    def find_closest_centroid(self, centroid, X):
        """
        Assigns each data point to the nearest centroid

        inputs:
            centroid: holds the coordinates of centroids
            X: given data points
        """
        assigned_centroid = []
        for i in X:
            distance = []
            for c in centroid:
                distance.append(self.calc_dist(i, c))
            assigned_centroid.append(np.argmin(distance))
        return assigned_centroid
    
    def calc_centroid(self, clusters, X):
        """
        Takes an average of all the data points of each centroid and moves the centroid to that average

        inputs:
            clusters: array of cluster names(numbers) associated with data points
            X: given data points
        """
        new_centroids = []
        new_data = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])], axis=1)
    
        for c in set(new_data['cluster']):
            current_cluster = new_data[new_data['cluster'] == c][new_data.columns[:-1]]
            cluster_mean = current_cluster.mean(axis = 0)
            new_centroids.append(cluster_mean)
    
        return new_centroids
    
    def write_to_csv(self, data):
        """
        Writes given data to result.csv file

        inputs:
            data: given data that user wants to write csv
        """
        with open('result.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=";")
            for x in range(len(data)):
                writer.writerow([x+1, data[x]])
    
    def calc_entrophy(self, cluster_count, n_sample):
        """
        Calculates entropy of clusters
        example:
            cluster1 has 15 samples
            cluster2 has 25 samples
            overall is 40
            entropy = -(15/40 * log2(15/40) + 25/40 * log2(25/40))

        inputs:
            cluster_count: type of dictionary that holds different cluster indexes as key and
            number of them as value
            n_sample: number of samples
        """
        sum = 0
        for key, value in cluster_count.items():
            sum = sum + (value/n_sample * log2(value/n_sample))
    
        entropy = -(sum)
        return round(entropy, 6)

    def k_plus(self, data, k):
        """
        initializes the centroids for K-means++

        inputs:
            data: array of data points having shape (100, 2)
            k: number of clusters
        """
        # initialize the centroids list and add a randomly selected data point to the list
        centroids = []
        centroids.append(data[np.random.randint(data.shape[0]), :])

        # compute remaining k - 1 centroids
        for c in range(k - 1):
            # initializes a list to store distances of data points from nearest centroid
            dist = []
            for i in range(data.shape[0]):
                point = data[i, :]
                d = sys.maxsize
    
                # compute distance of point from each of the previously selected centroid and store the minimum distance
                for j in range(len(centroids)):
                    temp_dist = self.calc_dist(point, centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)
    
            # select data point with maximum distance as our next centroid
            dist = np.array(dist)
            next_centroid = data[np.argmax(dist), :]
            centroids.append(next_centroid)
        return centroids
    
    
    def k_means_algorithm(self, data, centroids):
        """
        K-means algorithm to cluster the given data with received centroids from k-means++ algorithm

        inputs:
            data: given data set
            centroids: initialized centroids that received from k-means++
        """
        centroids = np.array(centroids)
    
        for i in range(10):
            get_centroids = self.find_closest_centroid(centroids, data)
            centroids = self.calc_centroid(get_centroids, data)

        # plt.figure()
        # plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
        # plt.scatter(data[:, 0], data[:, 1], alpha=0.1)
        # plt.show()
        get_centroids = self.find_closest_centroid(centroids, data)
    
        # for x in range(len(get_centroids)):
        #     print("{}: {}" .format(x+1, get_centroids[x]))
    
        self.write_to_csv(get_centroids)
        cluster_count = Counter(get_centroids)
        entropy = self.calc_entrophy(cluster_count, len(get_centroids))

        # NOTE: to see how the data are clustered uncomment the lines 183 ~ 186

        # categories = np.array(get_centroids)
        # colormap = np.array(['r', 'g', 'b', 'm'])
        # plt.scatter(data[:, 0], data[:, 1], c=colormap[categories])
        # plt.show()


        return entropy
    
        
    def run(self, k_value):
        
        data = self.load_data("2d.data").__array__()
        self.plot_data(data)
        print()

        k_diff_clusters = {}
        entropies = []
        
        for k in range(2, k_value+1):
            for i in range(32):
                centroids = self.k_plus(data, k)
                entropy = self.k_means_algorithm(data, centroids)
                entropies.append(entropy)
                # print("k #{}: entropy = {}".format(k, entropy))
            diff_cluster = Counter(entropies)
            print("Entropy with 32 run --> (entropy : amount of clusters with this entropy)")
            print(diff_cluster)
            print("For k = {} ---> Different clusters is {}".format(k, len(diff_cluster)))
            k_diff_clusters[k] = len(diff_cluster)
            entropies = []
            print("\n")
    
    

    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
