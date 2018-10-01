from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""

        values = numpy.array(values)
        X0 = numpy.ones((len(features), 1))
        X = numpy.array(features)
        Xnew = numpy.hstack((X0, X))
        Xnew_t = Xnew.transpose()
        self.weights = numpy.matmul(numpy.matmul(numpy.linalg.pinv(numpy.matmul(Xnew_t, Xnew)), Xnew_t), values)
        return self

        raise NotImplementedError


    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""

        weights = self.get_weights()
        bias = weights[0]
        weights_vector = numpy.array(weights[1:])
        predict_val = []
        for feature in features:
            p = numpy.matmul(weights_vector.transpose(), numpy.array(feature)) + bias
            predict_val.append(p)
        return predict_val

        raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.tolist()

        raise NotImplementedError


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""

        values = numpy.array(values)
        X0 = numpy.ones((len(features), 1))
        X = numpy.array(features)
        Xnew = numpy.hstack((X0, X))
        Xnew_t = Xnew.transpose()

        self.weights = numpy.matmul(
            numpy.matmul(
            numpy.linalg.inv(
            numpy.matmul(
            Xnew_t, Xnew)
            + self.alpha * numpy.identity(len(Xnew_t))
            ), Xnew_t
            ), values)

        return self

        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""

        weights = self.get_weights()
        bias = weights[0]
        weights_vector = numpy.array(weights[1:])
        predict_val = []
        for feature in features:
            p = numpy.matmul(weights_vector.transpose(), numpy.array(feature)) + bias
            predict_val.append(p)
        return predict_val

        raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights.tolist()

        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
