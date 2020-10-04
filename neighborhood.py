#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Neighborhood based Classifier
'''
from scipy.spatial import distance
import random


class NeighborhoodClassifier:
    def __init__(self, X, y):
        self.X = X              # data features
        self.y = y              # labels

    def find_neighborhood(self, x_test, R):

        neighborhood = []
        for xi, yi in zip(self.X, self.y):
            dist = distance.euclidean(xi, x_test)
            if dist <= R:                       # lies inside the circular neighborhood
                if dist != 0.0:                 # skipping distance from itself
                    neighborhood.append((dist, yi))

        neighborhood.sort(key=lambda tup: tup[0])  # sorting the distance in ascending order
        return neighborhood

    def predict(self, x_test, R):
        neighbors = self.find_neighborhood(x_test, R)          # Finds all the neighbors in the circular neighborhood
        # get the most frequent label class from nearest neighbors
        output_labels = list(dict(neighbors).values())
        if len(output_labels) == 0:                         # there are no neighbors in the neighborhood
            predicted_label = random.randint(0,1)           # making a random prediction
        else:
            predicted_label = max(set(output_labels), key=output_labels.count)
        return predicted_label
