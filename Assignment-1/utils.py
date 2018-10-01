from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    np.array(y_true)
    error_list = np.array(y_true) - np.array(y_pred)
    return np.mean(list(map(lambda error: error * error, error_list)))
    raise NotImplementedError


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    tp, tn, fp, fn = 0, 0, 0, 0
    for r, p in zip(real_labels, predicted_labels):
        if r == p:
            if r == 1:
                tp += 1
            else:
                tn += 1
        else:
            if r < p: # r:0; p:1
                fp += 1
            else:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
    raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    result = []
    for sample in features:
        sample_new = []
        i = 1
        while i <= k:
            sample_new += np.power(np.array(sample), i).tolist()
            i += 1
        result.append(sample_new)
    return result
    raise NotImplementedError



def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.sqrt(sum(np.square(np.array(point1) - np.array(point2))))
    raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return sum(np.array(point1) * np.array(point2))
    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    return -np.exp(-0.5 * sum(np.square(np.array(point1) - np.array(point2))))
    raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        result = []
        for sample in features:
            den = np.sqrt(sum(np.power(np.array(sample), 2)))
            if den != 0:
                tmp = list(map(lambda num: num/den, sample))
            else:
                tmp = sample
            result.append(tmp)
        return features
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        features_t = np.array(features).transpose().tolist()
        max_list = list(map(lambda l: max(l), features_t))
        min_list = list(map(lambda l: min(l), features_t))
        min_max = np.array([max_list, min_list]).transpose().tolist()
        result = []
        for sample in features:
            i = 0
            sample_new = []
            while i < len(sample):
                den = min_max[i][0] - min_max[i][1]
                if den != 0:
                    sample_new.append((sample[i] - min_max[i][1])/den)
                # 分母为零怎么处理
                else:
                    sample_new.append(sample[i])
                i += 1
            result.append(sample_new)
        return result

        raise NotImplementedError
