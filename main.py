import FileReader
import DataUtility
from KNearestNeighbor import KNearestNeighbor
import numpy as np
import DataVisualizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Use FileReader to import the training and testing data
    train_data = FileReader.import_data(
        "/Users/miguelschulz/PycharmProjects/pythonProject/training.txt"
    )
    test_data  = FileReader.import_data(
        "/Users/miguelschulz/Documents/Private Projects .nosync/Semesterprojekt Uni ML/Data/testing.txt"
    )

    # Slice array to separate features and classification
    train_x, train_y = DataUtility.extract_features_and_labels(train_data)#,[1,2,3,4,5,6,7,9,10,11,12])
    test_x, test_y = DataUtility.extract_features_and_labels(test_data)#,[1,2,3,4,5,6,7,8,9,10,11,12])

    # Filter for some basic criteria
    all_infected = train_data[np.where(train_data[:, 0] == 1)]
    all_healthy = train_data[np.where(train_data[:, 0] == -1)]
    all_males = train_data[np.where(train_data[:, 2] == 1)]
    all_females = train_data[np.where(train_data[:, 2] == 0)]

    # Print statistics in console
    DataVisualizer.analyze_basic_data(train_data)
    DataVisualizer.analyze_age(all_healthy, all_infected)
    DataVisualizer.analyze_weight(all_healthy, all_infected)
    DataVisualizer.analyze_by_gender(all_males, all_females)

    # Normalize values to fit in range 0 and 1
    # This ensures that np.bincount can be used to count labels after prediction
    # and makes sure all features are weighted equally
    train_x = DataUtility.normalize_by_column(train_x)
    test_x  = DataUtility.normalize_by_column(test_x)
    train_y = DataUtility.normalize_by_column(train_y)
    test_y  = DataUtility.normalize_by_column(test_y)

    features_infected = train_x[np.where(train_data[:, 0] == 1)]
    features_healthy = train_x[np.where(train_data[:, 0] == -1)]

    # Uncomment to plot comparison of features for infected and healthy
    # DataVisualizer.compare_boxplots(features_infected, features_healthy)

    # find what k to use for model
    k = DataUtility.cross_validation(train_x, train_y, show_graph=True)
    # set to True to see accuracy plotted for every cross validation split

    KNN = KNearestNeighbor(k) # init model with k from cross validation
    KNN.train(train_x, train_y) # train model with training data

    y_pred = KNN.predict(test_x) # predict the test data to get final accuracy of model
    final_accuracy = sum(y_pred == test_y) / test_y.shape[0]

    DataVisualizer.print_spacer()
    print(f"Model reached accuracy of {final_accuracy*100} % for predicting unknown data")

    accuracy_over_k = np.empty((0, 2), float)  # array for the accuracy per k
    for a in range(1, 32, 2):  # for every possible k in 1 to 31
        knn = KNearestNeighbor(k=a)
        knn.train(train_x, train_y)
        y_pred = knn.predict(test_x)
        accuracy_over_k = np.append(accuracy_over_k, np.array([[a, sum(y_pred == test_y) / test_y.shape[0]]]), axis=0)

    plt.plot(accuracy_over_k[:, 0], accuracy_over_k[:, 1])

    plt.xlabel("k")
    plt.ylabel("Accuracy")

    plt.show()