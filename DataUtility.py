from KNearestNeighbor import KNearestNeighbor
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

all_features = [1,2,3,4,5,6,7,8,9,10,11,12]
cutout_to_high = [1,2,3,4,5,6,7,9,11,12]
selected_features = [4,5,6]

def extract_features_and_labels(data, include_features = selected_features):
    return data[:, include_features], data[:, 0]

def avg_and_std(data):
    return np.average(data), np.std(data)

def min_and_max(data):
    return np.min(data), np.max(data)

def normalize_by_column(features):
    return (features - features.min(0)) / features.ptp(0)

def cross_validation(X, Y, show_graph=False, n_splits = 4):
    kf = KFold(n_splits) # init sklearn KFold
    accuracies_per_split = [] # array for the accuracies per k for all splits

    for train_index, test_index in kf.split(X): # iterate over possible split
        # get splitted data
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        accuracies_per_split.append(accuracy_over_k(X_train, Y_train, X_test, Y_test))

    if show_graph:
        fig, ax = plt.subplots(1, n_splits) # prepare subplots
        for i in range(0, n_splits):
            # prepare format that can be plotted
            x_values = accuracies_per_split[i][:, 0]
            y_values = accuracies_per_split[i][:, 1]

            ax[i].plot(x_values, y_values)

            ax[i].title.set_text(f"Split {i}")
            ax[i].set_xlabel("k")
            ax[i].set_ylabel("Accuracy")

        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.4,
            hspace=0.4
        )
        plt.show()

    best_ks = []
    for i in range(0, n_splits):
        maxIndex = np.argmax(accuracies_per_split[i], axis=0)[1]
        best_ks.append(accuracies_per_split[i][maxIndex, 0])

    occurrences = np.bincount(best_ks)  # count number of occurrences for every k
    return np.argmax(occurrences)  # find what k is most common


def accuracy_over_k(train_x, train_y, test_x, test_y):
    accuracy_over_k = np.empty((0, 2), float) # array for the accuracy per k
    for a in range(1, 32, 2):  # for every possible k in 1 to 31
        # train and predict with given data
        KNN = KNearestNeighbor(k=a)
        KNN.train(train_x, train_y)
        y_pred = KNN.predict(test_x)

        # calculate accuracy of prediction and append to array
        accuracy_over_k = np.append(accuracy_over_k, np.array([[a, sum(y_pred == test_y) / test_y.shape[0]]]), axis=0)

    return accuracy_over_k