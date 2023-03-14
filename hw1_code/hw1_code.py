import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.utils import shuffle
from scipy import sparse


def compute_information_gain(data, features, labels, word):
    data = data.toarray()
    index = -1
    split1 = []
    split1labels = []
    split2 = []
    split2labels = []
    numberoflineswiththe = 0
    numberoflineswithoutthe = 0
    num_real = 0
    num_total = 0
    for i in labels:
        if i > 0:
            num_real += 1
        num_total += 1
    for i in range(0, len(features)):
        if features[i] == word:
            index = i
    for i in range(0, len(data)):
        if data[i][index] > 0:
            split1.append(data[i])
            split1labels.append(labels[i])
            numberoflineswiththe += 1
        else:
            split2.append(data[i])
            split2labels.append(labels[i])
            numberoflineswithoutthe += 1
    #we can calculate new entropy with this. first lets calculate old entropy.
    p = num_real/num_total
    entropy_before = -p * math.log2(p) - (1-p) * math.log2(1-p)
    #now we calculate new entropy which is the weighted sum of the entropies of the two splits
    weight1 = numberoflineswiththe / (numberoflineswiththe + numberoflineswithoutthe)
    weight2 = numberoflineswithoutthe / (numberoflineswiththe + numberoflineswithoutthe)
    num_real = 0
    num_total = 0
    for i in split1labels:
        if i > 0:
            num_real += 1
        num_total += 1
    p1 = num_real/num_total
    num_real = 0
    num_total = 0
    for i in split2labels:
        if i > 0:
            num_real += 1
        num_total += 1
    p2 = num_real / num_total
    entropy_after = weight1 * (-p1 * math.log2(p1) - (1-p1) * math.log2(1-p1)) + weight2 * (-p2 * math.log2(p2) - (1-p2) * math.log2(1-p2))
    print("the word to split by is \"" + word + "\"")
    print("entropy before = " + str(entropy_before))
    print("entropy_after = " + str(entropy_after))
    print("therefore, the information gain is the entropy before the split minus the entropy after the split which is " + str(entropy_before-entropy_after))




def load_data():
    f = open("clean_fake.txt", "r")
    text1 = f.read().splitlines()
    len1 = len(text1)
    f = open("clean_real.txt", "r")
    text2 = f.read().splitlines()
    len2 = len(text2)
    text = text1 + text2
    vectorizer = CountVectorizer()
    text = vectorizer.fit_transform(text)
    label = np.hstack(([0] * len1, [1] * len2))  # 1 is real, 0 is fake
    training_data, x_other, training_labels, other_labels = train_test_split(text, label, test_size=0.30,
                                                                             random_state=random.seed(random.random()),
                                                                             shuffle=True, stratify=label)
    test_data, validation_data, test_labels, validation_labels = train_test_split(x_other, other_labels, test_size=0.50,
                                                                                  random_state=random.seed(
                                                                                      random.random()), shuffle=True,
                                                                                  stratify=other_labels)  # 50% of 30% is 15%
    return training_data, training_labels, test_data, test_labels, validation_data, validation_labels, vectorizer.get_feature_names_out(), text, label


def select_model(training_data, training_labels, test_data, test_labels, validation_data, validation_labels):
    squaremeanloss = []
    depths = []
    for i in range(5, 10):
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=i)
        clf = clf.fit(training_data, training_labels)
        predictions = []
        num = 0
        for x in validation_data:
            predictions.append(clf.predict(x))
        for x in range(len(predictions)):
            num += (predictions[x] - validation_labels[x]) * (predictions[x] - validation_labels[x]) / 2
        print(str(num) + " gini, depth:" + str(i))  # upon running the code, we see that gini with depth 9 produces
        # the lowest error consistently, hence the following code which is for depth 9 only in the gini section.
        if i == 9:
            test = []
            correct = 0
            for x in test_data:
                test.append(clf.predict(x))
            for x in range(len(test)):
                if test[x] - test_labels[x] == 0:
                    correct += 1
            print("^this tree is correct^ " + str(correct * 100 / len(test)) + "% of the time on the test data")
            fig = plt.figure(figsize=(15, 10))
            _ = tree.plot_tree(clf, max_depth=2, feature_names=feature_names, filled=True)
            plt.show()

        squaremeanloss.append(num)
        depths.append(i)
    # plt.plot(depths, squaremeanloss)
    # plt.xlabel('depth of tree')
    # plt.ylabel('mean square loss = (1/2)(y-t)^2')
    # plt.title('gini')
    # plt.show()

    for i in range(5, 10):
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=i)
        clf = clf.fit(training_data, training_labels)
        predictions = []
        num = 0
        for x in validation_data:
            predictions.append(clf.predict(x))
        for x in range(len(predictions)):
            num += (predictions[x] - validation_labels[x]) * (predictions[x] - validation_labels[x]) / 2
        print(str(num) + " entropy, depth:" + str(i))
    for i in range(5, 10):
        clf = tree.DecisionTreeClassifier(criterion="log_loss", max_depth=i)
        clf = clf.fit(training_data, training_labels)
        predictions = []
        num = 0
        for x in validation_data:
            predictions.append(clf.predict(x))
        for x in range(len(predictions)):
            num += (predictions[x] - validation_labels[x]) * (predictions[x] - validation_labels[x]) / 2
        print(str(num) + " log_loss, depth:" + str(i))


training_data, training_labels, test_data, test_labels, validation_data, validation_labels, feature_names, full_data, full_labels = load_data()
select_model(training_data, training_labels, test_data, test_labels, validation_data, validation_labels)
compute_information_gain(full_data, feature_names, full_labels, "the") #you can also pass it training data but then you have to switch feature_names to be the training data names
