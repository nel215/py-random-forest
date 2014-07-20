#coding:utf-8

from py_rf.decision_tree import DecisionTree
import requests
import random
import sys
import operator

def test_create_decision_tree():
    tree = DecisionTree()


def test_predict():
    dataset = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data').text.split('\n')
    dataset = filter(lambda data: len(data) > 1, dataset)
    dataset = map(lambda data: data.split(','), dataset)

    split = 2*len(dataset)/3
    trial_count = 50
    correct_ratio = 0

    for _ in xrange(trial_count):
        random.shuffle(dataset)
        train_data = dataset[:split]
        test_data  = dataset[split:]

        features = map(lambda data: map(float, data[:-1]), train_data)
        labels = map(lambda data: data[-1], train_data)

        tree = DecisionTree(max_depth=5)
        tree.build(features, labels)

        correct = 0
        for data in test_data:
            feature = map(float, data[:-1])
            label = data[-1]
            probability = tree.predict(feature)
            maxlabel = max(probability.iteritems(), key=operator.itemgetter(1))[0]
            correct += 1.0 if label == maxlabel else 0.0
        correct_ratio += 100.0 * correct / len(test_data)
    correct_ratio /= trial_count
    print correct_ratio
    assert correct_ratio >= 80.0, "sometime fial."
