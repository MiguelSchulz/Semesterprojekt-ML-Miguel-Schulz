import numpy as np
import DataUtility
import matplotlib.pyplot as plt
from KNearestNeighbor import KNearestNeighbor

def print_spacer():
    print("-----------------------------------------")

def analyze_basic_data(train_data):
    print(f"The average age is {np.average(train_data[:, 1])}")
    print(f"The average weight is {np.average(train_data[:, 3])}")

    print_spacer()

    all_infected = train_data[np.where(train_data[:, 0] == 1)]
    all_healthy = train_data[np.where(train_data[:, 0] == -1)]
    all_males = train_data[np.where(train_data[:, 2] == 1)]
    all_females = train_data[np.where(train_data[:, 2] == 0)]

    print(f"{len(all_healthy)} healthy and {len(all_infected)} infected in Training Data")
    print(f"{len(all_males)} are male and {len(all_females)} are female")

def analyze_age(all_healthy, all_infected):
    infected_column = all_infected[:, 1]
    healthy_column = all_healthy[:, 1]
    infected_average_age, infected_std_age = DataUtility.avg_and_std(infected_column)
    healthy_average_age, healthy_std_age = DataUtility.avg_and_std(healthy_column)

    infected_min_age, infected_max_age = DataUtility.min_and_max(infected_column)
    healthy_min_age, healthy_max_age = DataUtility.min_and_max(healthy_column)

    print_spacer()

    print(f"The average age of an infected person is {infected_average_age}")
    print(f"The average age of a healthy person is {healthy_average_age}")

    print(f"The std of age of an infected person is {infected_std_age}")
    print(f"The std of age of a healthy person is {healthy_std_age}")

    print_spacer()

    print(f"The youngest infected person is {infected_min_age}")
    print(f"The oldest infected person is {infected_max_age}")

    print(f"The youngest healthy person is {healthy_min_age}")
    print(f"The oldest healthy person is {healthy_max_age}")

    print_spacer()
    infected_under_18 = all_infected[np.where(all_infected[:, 1] < 18)]
    infected_over_18 = all_infected[np.where(all_infected[:, 1] >= 18)]

    print(f"{len(infected_under_18)} children under 18 are infected and {len(infected_over_18)} adults are infected")
    print(f"That means {len(infected_under_18) / len(all_infected) * 100} % of infected are children under 18")


def analyze_weight(all_healthy, all_infected):
    infected_column = all_infected[:, 3]
    healthy_column = all_healthy[:, 3]

    infected_averweight_weight, infected_std_weight = DataUtility.avg_and_std(infected_column)
    healthy_averweight_weight, healthy_std_weight = DataUtility.avg_and_std(healthy_column)

    infected_min_weight, infected_max_weight = DataUtility.min_and_max(infected_column)
    healthy_min_weight, healthy_max_weight = DataUtility.min_and_max(healthy_column)

    print_spacer()

    print(f"The average weight of an infected person is {infected_averweight_weight}")
    print(f"The average weight of a healthy person is {healthy_averweight_weight}")

    print(f"The std of weight of an infected person is {infected_std_weight}")
    print(f"The std of weight of a healthy person is {healthy_std_weight}")

    print_spacer()

    print(f"The lowest weight of an infected person is {infected_min_weight}")
    print(f"The highest weight of an infected person is {infected_max_weight}")

    print(f"The lowest weight of a healthy person is {healthy_min_weight}")
    print(f"The highest weight of a healthy person is {healthy_max_weight}")


def analyze_by_gender(male, female):
    print_spacer()

    all_infected_males = male[np.where(male[:, 0] == 1)]
    all_healthy_males = male[np.where(male[:, 0] == -1)]

    all_infected_females = female[np.where(female[:, 0] == 1)]
    all_healthy_females = female[np.where(female[:, 0] == -1)]

    count_infected_males = len(all_infected_males)
    count_healthy_males = len(all_healthy_males)

    count_infected_females = len(all_infected_females)
    count_healthy_females = len(all_healthy_females)

    print(f"There are {count_healthy_males} healthy males and {count_infected_males} infected males, which "
          f"mean {count_infected_males / len(male) * 100}% of males are infected")

    print(f"There are {count_healthy_females} healthy females and {count_infected_females} infected females, which "
          f"mean {count_infected_females / len(female) * 100}% of females are infected")

def boxplot_features(data):
    plt.boxplot(data)

    ax = plt.gca()
    ax.set_ylim([-60, 120])

    plt.show()

def compare_boxplots(data1, data2):
    fig, ax = plt.subplots(1, 2)
    ax[0].boxplot(data1)
    ax[1].boxplot(data2)

    ax[0].title.set_text("Infiziert")
    ax[1].title.set_text("Nicht-infiziert")
    plt.show()

def plot_all_testing_ks(train_x, train_y, test_x, test_y):
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
