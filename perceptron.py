#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Perceptron classifier
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)


def train_test_split(X, y, test_size):

    df = pd.DataFrame(X, columns=['P', 'N'])
    df['label'] = y
    # Shuffle dataset
    shuffle_df = df.sample(frac=1)

    # Define a size for your train set
    train_size = int((1-test_size) * len(df))

    # Split your dataset
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]

    X_train = train_set.to_numpy()[:, 0:-1]
    y_train = train_set.to_numpy()[:, -1]
    X_test = test_set.to_numpy()[:, 0:-1]
    y_test = test_set.to_numpy()[:, -1]

    return X_train, X_test, y_train, y_test


def balanced_acc(y_true, y_pred):

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


class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None

    def threshold_function(self, x):
        fx = np.dot(self.w, x)+self.b
        return 1 if fx > 0 else 0

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.threshold_function(x))

        return y_pred

    def fit(self, X, y, epochs=100, alpha=0.1, validation_split=0.2, weight_init='random',
            epoch_checkpoint=5, verbose=True, do_plot=True):

        # train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= validation_split)

        # weight initialization
        if weight_init == 'random':
            self.w = np.random.rand(X.shape[1])      # random init of weights to braak symmetry
        if weight_init == 'zeros':
            self.w = np.zeros(X.shape[1])
        if weight_init == 'ones':
            self.w = np.ones(X.shape[1])

        self.b = 0
        weights_history = []
        bias_history = []
        train_errors = {}
        val_errors = {}
        for i in range(epochs):
            for xi, yi in zip(X_train, y_train):
                y_pred = self.threshold_function(xi)
                # weight update
                self.w = self.w + (alpha * (yi - y_pred)) * xi
                # bias update
                self.b = self.b + (alpha * (yi - y_pred)) * 1

            weights_history.append(self.w)
            bias_history.append(self.b)

            if i % epoch_checkpoint == 0:        # checkpoint to track error
                _y_train_preds = self.predict(X_train)
                _y_val_preds = self.predict(X_val)
                train_err_i = 1 - balanced_acc(y_train, _y_train_preds)
                val_err_i = 1 - balanced_acc(y_val, _y_val_preds)
                train_errors[i] = train_err_i
                val_errors[i] = val_err_i

                if verbose:
                    print("Epoch %d: \n\t Training Error: %0.3f \t Validation Error: %0.3f"%(i, train_err_i,
                                                                                                   val_err_i))

        if do_plot:
            plt.plot(list(train_errors.keys()), list(train_errors.values()), label='E_train')
            plt.plot(list(val_errors.keys()), list(val_errors.values()), label='E_test')
            plt.xlabel('epochs')
            plt.ylabel('Error (1- balanced acc')
            plt.title('Training and Test Error of Perceptron with epochs')
            plt.legend(loc="upper left")
            # plt.xticks(list(train_errors.keys()))
            plt.savefig('figs/perceptron.pdf')
            plt.clf()
