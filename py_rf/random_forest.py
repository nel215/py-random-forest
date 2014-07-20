#coding: utf-8

from decision_tree import DecisionTree
import random
import requests
import numpy

class RandomForest:
    def __init__(self, num_tree=10):
        self.num_tree = num_tree
        self.attribute_indices_list = []
        self.trees = [DecisionTree(max_depth=6) for i in xrange(self.num_tree)]


    def build(self, features, labels):
        num_row = len(features)
        num_col = len(features[0])

        datum_indices = [i for i in xrange(num_row)]
        attribute_indices = [i for i in xrange(num_col)]
        for tree in self.trees:
            # select data
            random.shuffle(datum_indices)
            select_datum_indices = datum_indices[:2*num_row/3]
            select_features = numpy.take(features, select_datum_indices, axis=0).tolist()
            select_labels = numpy.take(labels, select_datum_indices, axis=0).tolist()

            # select attributes
            random.shuffle(attribute_indices)
            select_attribute_indices = attribute_indices[:2*num_col/3]
            select_features = numpy.take(select_features, select_attribute_indices, axis=1).tolist()
            self.attribute_indices_list.append(select_attribute_indices)

            tree.build(select_features, select_labels)


    def predict(self, feature):
        probabilities = {}
        for i, tree in enumerate(self.trees):
            select_attribute_indices = self.attribute_indices_list[i]
            select_feature = numpy.take(feature, select_attribute_indices, axis=0).tolist()
            probability = tree.predict(select_feature)
            for label, p in probability.items():
                if not probabilities.has_key(label): probabilities[label] = 0
                probabilities[label] += 1.0*p/self.num_tree
        return probabilities


if __name__=='__main__':
    import operator
    dataset = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data').text.split('\n')
    dataset = filter(lambda data: len(data) > 1, dataset)
    dataset = map(lambda data: data.split(','), dataset)

    split = 2*len(dataset)/3
    trial_count = 10
    correct_ratio = 0

    for _ in xrange(trial_count):
        random.shuffle(dataset)
        train_data = dataset[:split]
        test_data  = dataset[split:]

        features = map(lambda data: map(float, data[:-1]), train_data)
        labels = map(lambda data: data[-1], train_data)

        forest = RandomForest()
        forest.build(features, labels)

        correct = 0
        for data in test_data:
            feature = map(float, data[:-1])
            label = data[-1]
            probability = forest.predict(feature)
            maxlabel = max(probability.iteritems(), key=operator.itemgetter(1))[0]
            correct += 1.0 if label == maxlabel else 0.0
        correct_ratio += 100.0 * correct / len(test_data)
    correct_ratio /= trial_count
    print correct_ratio



    pass

