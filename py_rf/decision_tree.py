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
        self.attr = None
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


    def __init__(self, max_depth = 7):
        # TODO: gain function option
        # gini
        self.gain_function = lambda probabilities: 1.0 - numpy.sum(numpy.multiply(probabilities, probabilities))
        self.max_depth = max_depth
        pass

    def recursive(self, features, depth = 0):
        node = DecisionNode()

        classes = features.transpose()[-1]
        if depth is self.max_depth:
            node.is_leaf = True
            node.classes = numpy.bincount(classes.astype('int64'), minlength = self.num_class)
            return node

        # TODO: split multi pivot
        medians = map(lambda vector: numpy.median(vector), features.transpose()[0:-1])

        gains = []
        for attr, median in enumerate(medians):
            variables = features.transpose()[attr]
            lower_count = numpy.bincount(classes.take(numpy.where(variables <= median))[0].astype('int64'), minlength = self.num_class)
            upper_count = numpy.bincount(classes.take(numpy.where(variables >  median))[0].astype('int64'), minlength = self.num_class)

            lower_gain = self.gain_function(lower_count/(1.0*features.shape[0]))*numpy.sum(lower_count)
            upper_gain = self.gain_function(upper_count/(1.0*features.shape[0]))*numpy.sum(upper_count)
            gains.append(lower_gain + upper_gain)

        node.attr = numpy.argmax(gains)
        node.value = medians[node.attr]

        lower_select = numpy.where(features.transpose()[node.attr] <= node.value)[0].astype('int64')
        upper_select = numpy.where(features.transpose()[node.attr] >  node.value)[0].astype('int64')


        if lower_select.shape[0] is 0 or upper_select.shape[0] is 0:
            node.is_leaf = True
            node.classes = numpy.bincount(classes.astype('int64'), minlength = self.num_class)
            return node

        node.left_child  = self.recursive(features.take(lower_select, axis=0), depth + 1)
        node.right_child = self.recursive(features.take(upper_select, axis=0), depth + 1)

        return node


    def predict(self, feature):
        node = self.root
        while not node.is_leaf:
            node = node.left_child if feature[node.attr] <= node.value else node.right_child
        probability = {}
        for i, count in enumerate(node.classes):
            probability[self.class_to_label_dictionary[i]] = count/numpy.sum(node.classes)
        return probability


if __name__=='__main__':
    pass

