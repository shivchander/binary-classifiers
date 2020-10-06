#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
 Binary classification of Stressed/Not Stressed dataset using kNN, Neighborhood and Perceptron 
'''

import re
import numpy as np
from knn import KNN
from neighborhood import NeighborhoodClassifier
import matplotlib.pyplot as plt
from perceptron import Perceptron


def parse_reformat(txt_file):
    X = []
    y = []
    with open(txt_file, 'r') as f:
        for line in f:

            if 'Not Stressed' == line.lstrip().rstrip():
                label = 0
            if 'Stressed' == line.lstrip().rstrip():
                label = 1

            # checking if the line is of the form <float> \t <float>
            if re.search('[+-]?[0-9]+\.[0-9]+\t+[+-]?[0-9]+\.[0-9]+', line.lstrip().rstrip()):
                row = list(map(float, line.rstrip().split('\t')))
                X.append(row)
                y.append(label)

    return X, y


# Find the min and max values for each column
def dataset_minmax(X):
    minmax = list()
    for i in range(len(X[0])):
        col_values = [row[i] for row in X]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(X):
    minmax = dataset_minmax(X)
    for row in X:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return X


def balanced_acc(y_true, y_pred):
    """
    :param y_true: list of ground truth
    :param y_pred: list of predictions
    :return: balanced accuracy score
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    for y, y_hat in zip(y_true, y_pred):
        if y == y_hat == 1:
            tp += 1
        elif y == y_hat == 0:
            tn += 1
        elif y_hat == 1 and y == 0:
            fp += 1
        else:
            fn += 1
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    balanced_acc = (sensitivity + specificity) / 2

    return round(balanced_acc, 3)


def classifier(X, y, neighbor_param, classifier_type='knn'):
    """
    :param X: feature vector
    :param y: labels
    :param neighbor_param: k value for kNN or R value for Neighborhood
    :param classifier_type: kNN (default), Neighborhood
    :return: predictions
    """
    y_pred = []
    if classifier_type == 'knn':
        model = KNN(X, y)
    if classifier_type == 'neighborhood':
        model = NeighborhoodClassifier(X, y)

    for xi, y_true in zip(X, y):
        y_pred.append(model.predict(xi, neighbor_param))

    return y_pred


def q1(X, y):
    """
    :param X: Feature vectors of the dataset
    :param y: labels/ classes
    :return: None - Saves the two figures in the figs directory (should exist)
    """

    k_nn_performance = {}
    neighborhood_performance = {}
    for n in range(1, 12, 2):
        knn_predictions = classifier(X, y, n, 'knn')
        neighborhood_predictions = classifier(X, y, n, 'neighborhood')

        k_nn_performance[n] = round(balanced_acc(y, knn_predictions), 3)
        neighborhood_performance[n] = round(balanced_acc(y, neighborhood_predictions), 3)

    plt.plot(list(k_nn_performance.keys()), list(k_nn_performance.values()))
    plt.xlabel('k-value')
    plt.ylabel('Performance (Balanced Accuracy)')
    plt.title('Performance of kNN Classifier for different k-values')
    plt.xticks(list(k_nn_performance.keys()))
    plt.savefig('figs/knn_performance.pdf')
    plt.clf()

    plt.plot(list(neighborhood_performance.keys()), list(neighborhood_performance.values()))
    plt.xlabel('Radius')
    plt.ylabel('Performance (Balanced Accuracy)')
    plt.title('Performance of Neighborhood Classifier for different Radius')
    plt.xticks(list(neighborhood_performance.keys()))
    plt.savefig('figs/neighborhood_performance.pdf')
    plt.clf()


def q2(X, y):
    X = np.array(normalize_dataset(X))
    model = Perceptron()
    model.fit(X, y, alpha=0.001, weight_init='random', epochs=200, verbose=True, do_plot=True)


if __name__ == '__main__':
    X, y = parse_reformat('data/HW2_data.txt')
    # q1(X, y)
    q2(X, y)






