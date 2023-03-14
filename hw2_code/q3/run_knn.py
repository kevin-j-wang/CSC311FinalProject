from l2_distance import l2_distance
from hw2_code.q3.utils import *

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    k_values = [1, 3, 5, 7, 9] # x axis
    acc_values = [] # y axis
    for k in k_values:
        output = knn(k, train_inputs, train_targets, valid_inputs)
        accuracy = 0
        for i in range(0, len(output)):
            if output[i] == valid_targets[i]:
                accuracy += 1
        accuracy = accuracy/len(output)
        acc_values.append(accuracy)
    # plt.plot(k_values, acc_values)
    # plt.xlabel('k values')
    # plt.ylabel('accuracy')
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # plt.title('KNN, # of neighbors to accuracy chart')
    # plt.show()

    #ii
    output1 = knn(3, train_inputs, train_targets, valid_inputs)
    output2 = knn(3, train_inputs, train_targets, test_inputs)
    accuracy1 = 0
    accuracy2 = 0
    for i in range(0, len(output1)):
        if output1[i] == valid_targets[i]:
            accuracy1 += 1
        if output2[i] == test_targets[i]:
            accuracy2 += 1
    accuracy1 = accuracy1 / len(output1)
    accuracy2 = accuracy2 / len(output1)
    print(accuracy1)
    print(accuracy2)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
