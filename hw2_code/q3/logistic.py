from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    z = np.dot(data, weights[:-1]) + weights[-1:] #we dont apply sigmoid here because we need z not sigmoid(z) for calculations
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return z


def evaluate(targets, z):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    y = sigmoid(z)
    y = y.reshape(-1, 1)
    ce = np.mean(targets * np.logaddexp(0, -z).reshape(-1, 1) + (1 - targets) * np.logaddexp(0, z).reshape(-1, 1))
    y = np.where(y < 0.5, 0, 1)
    frac_correct = 1 - np.mean(np.absolute(y - targets))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    z = logistic_predict(weights, data)
    y = sigmoid(z)
    y = y.reshape(-1, 1)
    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    f = np.mean(targets * np.logaddexp(0, -z).reshape(-1, 1) + (1 - targets) * np.logaddexp(0, z).reshape(-1, 1))
    df = np.append(np.dot(data.T, (y - targets)) / len(targets), np.sum(y - targets) / len(targets))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
