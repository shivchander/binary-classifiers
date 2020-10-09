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
from perceptron import Perceptron, train_test_split
from statistics import mean, stdev


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


def metrics(y_true, y_pred):
    """
    :param y_true: list of ground truth
    :param y_pred: list of predictions
    :return: dict of metrics {metric: score}
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    metric = {}
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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    metric['precision'] = round(precision, 3)
    metric['recall'] = round(recall, 3)
    metric['balanced_acc'] = round(balanced_acc, 3)
    metric['f1'] = round(f1, 3)

    return metric


def classifier(X, y, neighbor_param, classifier_type='knn', X_test = []):
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

    if len(X_test) != 0:
        for xi in X_test:
            y_pred.append(model.predict(xi, neighbor_param))
    else:
        for xi, y_true in zip(X, y):
            y_pred.append(model.predict(xi, neighbor_param))

    return y_pred


def plot_metrics(classifier_metrics, title):
    fig, a = plt.subplots(2, 2)
    width = 0.35
    x = np.arange(1, 10)
    if title == 'Perceptron':
        a[0][0].bar(x - width/2, [i[0] for i in classifier_metrics['balanced_acc']], width, label='Train')
        a[0][0].bar(x + width/2, [i[1] for i in classifier_metrics['balanced_acc']], width, label='Test')
        a[0][0].legend()
    else:
        a[0][0].bar(x, classifier_metrics['balanced_acc'])
    a[0][0].set_xlabel('Trial #')
    a[0][0].set_ylabel('Balanced Accuracy')
    a[0][0].set_xticks(x)
    a[0][0].set_title(title+' Balanced Accuracy')
    if title == 'Perceptron':
        a[0][1].bar(x - width / 2, [i[0] for i in classifier_metrics['f1']], width, label='Train')
        a[0][1].bar(x + width / 2, [i[1] for i in classifier_metrics['f1']], width, label='Test')
        a[0][1].legend()
    else:
        a[0][1].bar(x, classifier_metrics['f1'])
    a[0][1].set_xlabel('Trial #')
    a[0][1].set_ylabel('f1 score')
    a[0][1].set_xticks(x)
    a[0][1].set_title(title + ' f1 score')
    if title == 'Perceptron':
        a[1][0].bar(x - width / 2, [i[0] for i in classifier_metrics['precision']], width, label='Train')
        a[1][0].bar(x + width / 2, [i[1] for i in classifier_metrics['precision']], width, label='Test')
        a[1][0].legend()
    else:
        a[1][0].bar(x, classifier_metrics['precision'])
    a[1][0].set_xlabel('Trial #')
    a[1][0].set_ylabel('precision')
    a[1][0].set_xticks(x)
    a[1][0].set_title(title + ' Precision')
    if title == 'Perceptron':
        a[1][1].bar(x - width / 2, [i[0] for i in classifier_metrics['recall']], width, label='Train')
        a[1][1].bar(x + width / 2, [i[1] for i in classifier_metrics['recall']], width, label='Test')
        a[1][1].legend()
    else:
        a[1][1].bar(x, classifier_metrics['recall'])
    a[1][1].set_xlabel('Trial #')
    a[1][1].set_ylabel('recall')
    a[1][1].set_xticks(x)
    a[1][1].set_title(title + ' Recall')
    plt.tight_layout()
    plt.savefig('figs/'+title+'_metrics.pdf')
    plt.clf()


def print_table(classifier_metrics, title):
    if title == 'Perceptron':
        precision = [i[1] for i in classifier_metrics['precision']]
        recall = [i[1] for i in classifier_metrics['recall']]
        f1 = [i[1] for i in classifier_metrics['f1']]
        balanced_acc = [i[1] for i in classifier_metrics['balanced_acc']]
    else:
        precision = classifier_metrics['precision']
        recall = classifier_metrics['recall']
        f1 = classifier_metrics['f1']
        balanced_acc = classifier_metrics['balanced_acc']

    print(title)
    print('\t', 'Balanced Accuracy: ', round(mean(balanced_acc), 3), '+/-', round(stdev(balanced_acc), 3))
    print('\t', 'F1 score: ', round(mean(f1), 3), '+/-', round(stdev(f1), 3))
    print('\t', 'Precision: ', round(mean(precision), 3), '+/-', round(stdev(precision), 3))
    print('\t', 'Recall: ', round(mean(recall), 3), '+/-', round(stdev(recall), 3))


def q1(X, y):
    """
    :param X: Feature vectors of the dataset
    :param y: labels/ classes
    :return: None - Saves the two figures in the figs directory (should exist)
    """
    print('Solving q1')
    k_nn_performance = {}
    neighborhood_performance = {}
    for n in range(1, 12, 2):
        knn_predictions = classifier(X, y, n, 'knn')
        neighborhood_predictions = classifier(X, y, n, 'neighborhood')

        k_nn_performance[n] = metrics(y, knn_predictions)['balanced_acc']
        neighborhood_performance[n] = metrics(y, neighborhood_predictions)['balanced_acc']

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
    print('Solving q2')
    model = Perceptron()
    model.fit(X, y, alpha=0.001, weight_init='random', epochs=200, verbose=False, do_plot=True)


def q3(X, y):                           # comparison of kNN, neighborhood, perceptron
    print('Solving q3')
    knn_performance = {'balanced_acc': [], 'precision': [], 'recall': [], 'f1': []}
    neighborhood_performance = {'balanced_acc': [], 'precision': [], 'recall': [], 'f1': []}
    perceptron_performance = {'balanced_acc': [], 'precision': [], 'recall': [], 'f1': []}
    perceptron_error = []
    for i in range(0, 9):
        # splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # testing and scoring kNN and neighborhood
        knn_predictions = classifier(X, y, 11, 'knn', X_test=X_test)
        neighborhood_predictions = classifier(X, y, 3, 'neighborhood', X_test=X_test)
        knn_performance['balanced_acc'].append(metrics(y_test, knn_predictions)['balanced_acc'])
        knn_performance['precision'].append(metrics(y_test, knn_predictions)['precision'])
        knn_performance['recall'].append(metrics(y_test, knn_predictions)['recall'])
        knn_performance['f1'].append(metrics(y_test, knn_predictions)['f1'])

        neighborhood_performance['balanced_acc'].append(metrics(y_test, neighborhood_predictions)['balanced_acc'])
        neighborhood_performance['precision'].append(metrics(y_test, neighborhood_predictions)['precision'])
        neighborhood_performance['recall'].append(metrics(y_test, neighborhood_predictions)['recall'])
        neighborhood_performance['f1'].append(metrics(y_test, neighborhood_predictions)['f1'])

        # testing and scoring perceptron
        model = Perceptron()
        perceptron_error.append(model.fit(X_train, y_train, alpha=0.001, weight_init='random', epochs=200,
                                          do_validation=False, training_error_track=True, epoch_checkpoint=10))
        y_train_preds = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        perceptron_performance['balanced_acc'].append([metrics(y_train, y_train_preds)['balanced_acc'],
                                                       metrics(y_test, y_test_pred)['balanced_acc']])
        perceptron_performance['precision'].append([metrics(y_train, y_train_preds)['precision'],
                                                    metrics(y_test, y_test_pred)['precision']])
        perceptron_performance['recall'].append([metrics(y_train, y_train_preds)['recall'],
                                                 metrics(y_test, y_test_pred)['recall']])
        perceptron_performance['f1'].append([metrics(y_train, y_train_preds)['f1'],
                                             metrics(y_test, y_test_pred)['f1']])

    # part a - metric plots
    print('Solving q3 - A')
    plot_metrics(knn_performance, 'kNN')
    plot_metrics(neighborhood_performance, 'Neighborhood')
    plot_metrics(perceptron_performance, 'Perceptron')

    # part b - printing table
    print('Solving q3 - B')
    print_table(knn_performance, 'kNN')
    print_table(neighborhood_performance, 'Neighborhood')
    print_table(perceptron_performance, 'Perceptron')

    # part c - trial-wise training error for perceptron
    print('Solving q3 - C')
    training_errors = []
    for i, e in enumerate(perceptron_error):
        plt.plot(list(e.keys()), list(e.values()), label='Trial #'+str(i+1))
        training_errors.append(list(e.values()))
    plt.xlabel('epochs')
    plt.ylabel('Error (1- balanced acc')
    plt.title('Trial-Wise Training Error Time-Series for the Perceptrons')
    plt.legend(loc="upper right")
    plt.xticks(list(perceptron_error[0].keys()))
    plt.savefig('figs/trialwise_error_perceptron.pdf')
    plt.clf()

    # part d - Perceptron mean training error
    print('Solving q3 - D')
    mean_errors = list(np.mean(training_errors, axis=0))
    std_errors = list(np.std(training_errors, axis=0))
    plt.errorbar(list(perceptron_error[0].keys()), mean_errors, std_errors)
    plt.xlabel('epochs')
    plt.ylabel('Error (1- balanced acc)')
    plt.title('Mean and Std Dev of Training Error for the Perceptrons')
    plt.xticks(list(perceptron_error[0].keys()))
    plt.savefig('figs/mean_error_perceptron.pdf')
    plt.clf()

    # decision boundaries
    # define bounds of the domain
    X = np.array(X)
    y = np.array(y)
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))
    grid_list = grid.tolist()
    # make knn predictions for the grid
    print('Solving q3 - E')
    yhat_knn = np.array(classifier(X, y, 11, 'knn', X_test=grid_list))
    # make neighborhood predictions for the grid
    print('Solving q3 - F')
    yhat_neighborhood = np.array(classifier(X, y, 3, 'neighborhood', X_test=grid_list))
    # make perceptron predictions for the grid
    print('Solving q3 - G')
    yhat_perceptron = np.array(model.predict(grid))

    # reshape the predictions back into a grid
    zz_knn = yhat_knn.reshape(xx.shape)
    zz_neighborhood = yhat_neighborhood.reshape(xx.shape)
    zz_perceptron = yhat_perceptron.reshape(xx.shape)

    # part e - knn decision boundary
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz_knn, cmap='Paired')
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
    plt.title('kNN Decision Boundary')
    plt.xlabel('P Range')
    plt.ylabel('N Range')
    plt.savefig('figs/knn_decision_boundary.pdf')
    plt.clf()

    # part f - neighborhood decision boundary
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz_neighborhood, cmap='Paired')
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
    plt.title('Neighborhood Decision Boundary')
    plt.xlabel('P Range')
    plt.ylabel('N Range')
    plt.savefig('figs/neighborhood_decision_boundary.pdf')
    plt.clf()

    # part f - perceptron decision boundary
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz_perceptron, cmap='Paired')
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
    plt.title('Perceptron Decision Boundary')
    plt.xlabel('P Range')
    plt.ylabel('N Range')
    plt.savefig('figs/perceptron_decision_boundary.pdf')
    plt.clf()


if __name__ == '__main__':
    import time
    start_time = time.time()
    X, y = parse_reformat('data/HW2_data.txt')
    q3(X, y)
    q1(X, y)
    q2(X, y)
    print("--- %s seconds ---" % (time.time() - start_time))
