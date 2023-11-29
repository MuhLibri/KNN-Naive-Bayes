import pandas as pd
from typing import TypeAlias, Callable
from enum import Enum
from math import sqrt
from functools import reduce
import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score
import implement

_MINKOWSKI_ORD_DEFAULT = 2.
_DISTANCE_CACHE: dict[int, tuple[list[float], float]] = dict()

def hash_row(li: list[float]) -> int:
    h = 0
    for f in li:
        h = (h << 1) ^ hash(f)
    h = (h << 1) ^ hash(_MINKOWSKI_ORD_DEFAULT)
    return h

# def dist_euclidean(v1: list[float], v2: list[float]) -> float:
#     sq_dist = sum([(v2[i] - v1[i]) ** 2 for i in range(len(v1) - 1)])
#     return sqrt(sq_dist)

# def dist_manhattan(v1: list[float], v2: list[float]) -> float:
#     return sum([abs(v2[i] - v1[i]) for i in range(len(v1) - 1)])

def dist_minkowski(v1: list[float], v2: list[float], ord: float = _MINKOWSKI_ORD_DEFAULT) -> float:
    return sum([abs(v2[i] - v1[i]) ** ord for i in range(len(v1) - 1)]) ** (1. / ord)

# def dist_gower(v1: list, v2: list, ranges: list[float], types: list[ColumnType] = _COLUMN_TYPES_DEFAULT) -> float:
#     similarity = 0.
#     for i in range(len(v1) - 1):
#         v = (
#             max(1. - dist_minkowski([v1[i]], [v2[i]]) / ranges[i], 0.)
#         ) if types[i] == ColumnType.QUANTITATIVE else (
#             1. if v1[i] == v2[i] else 0.
#         )
#         similarity += v
#     return sqrt(1. - similarity / (len(v1) - 1))

def knn(train, test, k):
    # norm_train = normalized(train)
    norm_train = train

    # Create initial list
    predictions = list()
    hit, miss = 0, 0
    
    # Iterate through all rows in the validation dataset
    for validation_row in test:
        # Create an empty list, soon to be filled with Euclidean distance of
        # corresponding rows between rows in the train dataset and the validation
        # dataset
        distances = list()

        # Iterate through train dataset
        for train_row in norm_train:
            # Check cache first and skip if the distance is already calculated
            h = hash_row(train_row[:-1] + validation_row[:-1])
            if h in _DISTANCE_CACHE:
                hit += 1
                distances.append(_DISTANCE_CACHE[h])
                continue

            # Append the distance (using the given distance function) to the distances list
            miss += 1
            # dist = (train_row, dist_gower(train_row, validation_row, ranges))
            dist = (train_row, dist_minkowski(train_row, validation_row))
            distances.append(dist)
            _DISTANCE_CACHE[h] = dist

            # # Calculate Euclidean distance between train dataset and validation dataset
            # distance = 0.0
            # for i in range(len(validation_row) - 1):
            #     distance += (validation_row[i] - train_row[i]) ** 2

            # # Append the Euclidean distance to the distances list
            # distances.append((train_row, sqrt(distance)))

        # Sort the distances list in ascending order
        distances.sort(key=lambda tup: tup[1])

        # Select the k highest items to be picked as neighbors
        neighbors = [dist[0] for dist in distances[:k]]

        # Find the majority class from k nearest neighbors = prediction
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)

        # Append the prediction result
        predictions.append(prediction)
    
    print(f'Cache hits = {hit}, misses = {miss}, hit ratio = {hit / (hit + miss)}')
    return predictions


def train_knn(df_train, df_validation):
    new_train = utils.convert_target_to_int(df_train)
    new_validation = utils.convert_target_to_int(df_validation)

    train_target = utils.get_target(new_train)
    validation_target = utils.get_target(new_validation)
    x_train = utils.get_x(new_train)
    x_validation = utils.get_x(new_validation)
    # new_train = utils.unpop(new_train, train_target)
    # new_validation = utils.unpop(new_validation, validation_target)

    classifier_result = knn(new_train, new_validation, 19)
    print(f'Result for K = 19:')
    print("Manual accuracy:", accuracy_score(validation_target, classifier_result))
    print("Manual precision: ", precision_score(validation_target, classifier_result, average='micro'))
    print("Manual recall: ", recall_score(validation_target, classifier_result, average='micro'))

    implement.determine_k(x_train, x_validation, train_target, validation_target)
    implement.knn_sklearn(x_train, x_validation, train_target, validation_target, 19)

    return classifier_result


def predict_knn(df_train, df_test):
    new_train = utils.convert_target_to_int(df_train)
    new_test = utils.convert_target_to_int(df_test)

    new_test = utils.exclude_id(new_test)
    # print(new_test)

    # minmax_train = find_minmax(new_train)
    # minmax_test = find_minmax(new_test)
    # normalization(new_train, minmax_train)
    # normalization(new_test, minmax_test)
    # print(new_train)
    # print(new_test)

    classifier_result = knn(new_train, new_test, 19)
    # print(classifier_result)

    return classifier_result


def column_extrema(data: list[list[float]]) -> list[tuple[float, float]]:
    return [(min(column), max(column)) for column in list(zip(*data))]

def normalized(data: list[list[float]]) -> list[list[float]]:
    extrema = column_extrema(data)
    norm_data: list[list[float]] = list()

    for row in data:
        norm_data.append([(row[i] - extrema[i][0]) / (extrema[i][1] - extrema[i][0]) for i in range(len(row) - 1)] + row[-1:])
    
    return norm_data

# def find_minmax(data):
#     minmax = list()
#     for i in range(len(data[0])):
#         col_values = [row[i] for row in data]
#         value_min = min(col_values)
#         value_max = max(col_values)
#         minmax.append([value_min, value_max])
#     return minmax
#
#
# def normalization(data, minmax):
#     for row in data:
#         for i in [12, 13, 14]:
#             row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])



