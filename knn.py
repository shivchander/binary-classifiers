#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
kNN Classifier
'''
from scipy.spatial import distance


class KNN:
    def __init__(self, X, y):
        self.X = X                              # data features
        self.y = y                              # labels

    def find_neighbor_distances(self, x_test):
        neighbors = []
        for xi, yi in zip(self.X, self.y):
            dist = distance.euclidean(xi, x_test)
            if dist != 0.0:                         # skipping distance from itself
                neighbors.append((dist, yi))

        neighbors.sort(key=lambda tup: tup[0])      # sorting the distance in ascending order
        return neighbors

    def predict(self, x_test, k):
        distances = self.find_neighbor_distances(x_test)          # calc the distance from all other points
        # get k nearest neighbors (least distances)
        k_neighbors = distances[:k]
        # get the most frequent label class from nearest neighbors
        output_labels = list(dict(k_neighbors).values())
        predicted_label = max(set(output_labels), key=output_labels.count)
        return predicted_label
