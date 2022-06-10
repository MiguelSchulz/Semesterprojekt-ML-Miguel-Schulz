import numpy as np


class KNearestNeighbor:
    def __init__(self, k):
        """
        Initializes a KNN classifier
        :param k: Odd number that sets how many of the closest neighbors should be used to predict from input features
        """
        self.k = k

    def train(self, X, y):
        """
        Sets the training data which should be used to predict labels
        :param X: All features for prediction
        :param y: The right labels
        """
        self.X_train = X
        self.y_train = y

    def _eucledian_distance(self, vector1, vector2):
        """
        Calculates the euclidean distance between between two rows of data
        :param vector1: n rows containing features
        :param vector2: A row containing features
        :return: Distance of vector2 to every row of data in vector1
        """
        return np.sqrt(
            np.sum((vector1 - vector2) ** 2, axis = 1) # sum along y axis
        )

    def predict(self, X_test):
        """
        Predict the label for every row of features passed based on training data
        :param X_test: All features that should be predicted
        :return: Predictions for given features
        """
        num_test = X_test.shape[0]  # number of samples to be predicted
        num_train = self.X_train.shape[0]  # number of training samples we have
        distances = np.zeros((num_test, num_train))  # init matrix that stores all distances

        for i in range(num_test):  # for every sample
            distances[i, :] = self._eucledian_distance(
                self.X_train,
                X_test[i, :]
            )  # calculate the eucledian distance to every training sample

        predict_y = np.zeros(num_test)  # init array that will hold the prediction for every sample

        for i in range(num_test):  # for every sample
            y_sorted_indices = np.argsort(distances[i, :])  # get indices of distances sorted in ascending order
            k_closest_labels = self.y_train[y_sorted_indices[: self.k]]\
                .astype(int)  # find the labels of the k closest training samples
            label_occurences = np.bincount(k_closest_labels)  # count number of occurences for each label
            predict_y[i] = np.argmax(label_occurences)  # find what label is more common

        return predict_y