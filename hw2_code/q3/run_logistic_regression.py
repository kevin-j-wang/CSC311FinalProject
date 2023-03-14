from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0.,
        "num_iterations": 2000
    }
    weights = np.append(np.ones(M)/1000, [0])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    #run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    cross_entropy_valid = []
    cross_entropy_train = []
    iteration = []
    counter = 1
    cross_entropy_valid.append(evaluate(valid_targets, logistic_predict(weights, valid_inputs))[0])
    cross_entropy_train.append(evaluate(train_targets, logistic_predict(weights, train_inputs))[0])
    iteration.append(counter)
    f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
    for t in range(hyperparameters["num_iterations"]):
        weights = weights - (hyperparameters["learning_rate"] * df)
        f2, df2 = logistic(weights, train_inputs, train_targets, hyperparameters)[:2]
        if (f - f2) < 0.000001:
            break
        else:
            f = f2
            df = df2
        cross_entropy_valid.append(evaluate(valid_targets, logistic_predict(weights, valid_inputs))[0])
        cross_entropy_train.append(evaluate(train_targets, logistic_predict(weights, train_inputs))[0])
        counter += 1
        iteration.append(counter)
    # plt.plot(iteration, cross_entropy_valid, "-b", label = "validation")
    # plt.plot(iteration, cross_entropy_train, "-r", label = "training")
    # plt.legend(loc="upper right")
    # plt.xlabel('iteration')
    # plt.ylabel('cross entropy')
    # plt.title('mnist_train_small')
    # plt.show()
    print(evaluate(train_targets, logistic_predict(weights, train_inputs)))
    print(evaluate(valid_targets, logistic_predict(weights, valid_inputs)))
    print(evaluate(test_targets, logistic_predict(weights, test_inputs))) #evaluating on test. spoiler alert its 92% which is higher than validation, which had 88%

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)
    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
