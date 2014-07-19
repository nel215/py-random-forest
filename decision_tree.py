#coding:utf-8

import numpy
import requests


def get_label_to_class_dictionary(labels):
    dictionary = {}
    for label in labels:
        if not dictionary.has_key(label): dictionary[label] = len(dictionary.keys())
    return dictionary


class DecisionNode:
    def __init__(self):
        self.is_leaf = False
        self.leff_child  = None
        self.right_child = None
        self.classes = None


class DecisionTree:
    def build(self, features, labels):
        self.label_to_class_dictionary = get_label_to_class_dictionary(labels)
        self.class_to_label_dictionary = {v:k for k,v in self.label_to_class_dictionary.items()}
        self.num_class = len(self.class_to_label_dictionary.keys())
        for i, c in enumerate( map(lambda label: self.label_to_class_dictionary[label], labels)):
            features[i].append(c)
        self.root = self.recursive(numpy.array(features))


    def __init__(self):
        # TODO: gain function option
        # TODO: depth option
        # gini
        self.gain_function = lambda probabilities: 1.0 - numpy.sum(numpy.multiply(probabilities, probabilities))
        pass

    def recursive(self, features, depth = 0):
        medians = map(lambda vector: numpy.median(vector), features.transpose()[0:-1])

        classes = features.transpose()[-1]
        gains = []
        for attr, median in enumerate(medians):
            variables = features.transpose()[attr]
            lower_count = numpy.bincount(classes.take(numpy.where(variables <= median))[0].astype('int64'), minlength = self.num_class)
            upper_count = numpy.bincount(classes.take(numpy.where(variables >  median))[0].astype('int64'), minlength = self.num_class)

            lower_gain = self.gain_function(lower_count/(1.0*features.shape[0]))*numpy.sum(lower_count)
            upper_gain = self.gain_function(upper_count/(1.0*features.shape[0]))*numpy.sum(upper_count)
            gains.append(lower_gain + upper_gain)

        select_attr = numpy.argmax(gains)

        lower_select = numpy.where(features.transpose()[select_attr] <= medians[select_attr])[0].astype('int64')
        upper_select = numpy.where(features.transpose()[select_attr] >  medians[select_attr])[0].astype('int64')

        node = DecisionNode()

        if lower_select.shape[0] is 0 or upper_select.shape[0] is 0:
            node.is_leaf = True
            node.classes = classes
            return node

        node.left_child = self.recursive(features.take(lower_select, axis=0), depth + 1)
        node.left_child = self.recursive(features.take(upper_select, axis=0), depth + 1)

        return node


if __name__=='__main__':
    dataset = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data').text.split('\n')
    dataset = filter(lambda data: len(data) > 1, dataset)
    dataset = map(lambda data: data.split(','), dataset)
    features = map(lambda data: map(float, data[:-1]), dataset)
    labels = map(lambda data: data[-1], dataset)

    tree = DecisionTree()
    tree.build(features, labels)
