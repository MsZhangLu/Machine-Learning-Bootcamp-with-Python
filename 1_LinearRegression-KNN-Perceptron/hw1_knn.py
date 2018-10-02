from __future__ import division, print_function

from typing import List, Callable


import numpy
import scipy




############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        # 对于每个点，相邻点距离排序
        # 前k个，数label个数
        # major label 定为这个点的label
        self.train_features = features
        self.train_labels = labels

        return self
        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        predicted_labels = []
        for feature1 in features:
            distance_tuples = []
            labels_name = set(self.train_labels)
            # predicted_labels = []
            for feature2, label in zip(self.train_features, self.train_labels):
                # LLO
                if feature1 != feature2:
                    d = self.distance_function(feature1, feature2)
                    distance_tuples.append((feature2, d, label))
            distance_sorted = sorted(distance_tuples, key = lambda distance: distance[1])
            knn = distance_sorted[:self.k] # [(feature, distance, label), ... ]
            knn_labels = list(map(lambda distance_tuple: distance_tuple[2], knn))
            knn_labels_soreted = [(label, knn_labels.count(label)) for label in labels_name]
            predicted_label = sorted(knn_labels_soreted, key = lambda l: l[1], reverse = True)[0][0]
            predicted_labels.append(predicted_label)
        return predicted_labels
        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
