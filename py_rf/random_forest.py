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
    pass

