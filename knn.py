import pandas as pd
from math import sqrt
import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score
import implement

def knn(train, test, k):
    # Create initial list
    predictions = list()
    
    # Iterate through all rows in the validation dataset
    for validation_row in test:
        # Create an empty list, soon to be filled with Euclidean distance of
        # corresponding rows between rows in the train dataset and the validation
        # dataset
        distances = list()

        # Iterate through train dataset
        for train_row in train:
            # Calculate Euclidean distance between train dataset and validation dataset
            distance = 0.0
            for i in range(len(validation_row) - 1):
                distance += (validation_row[i] - train_row[i]) ** 2

            # Append the Euclidean distance to the distances list
            distances.append((train_row, sqrt(distance)))

        # Sort the distances list in ascending order
        distances.sort(key=lambda tup: tup[1])

        # Create an empty list soon to be filled with neighbors: the closest k instances
        neighbors = list()

        # Select the k highest items to be picked as neighbors
        for i in range(k):
            neighbors.append(distances[i][0])

        # Find the majority class from k nearest neighbors = prediction
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)

        # Append the prediction result
        predictions.append(prediction)
    return predictions


def train_knn(df_train, df_validation):
    new_train = utils.convert_target_to_int(df_train)
    new_validation = utils.convert_target_to_int(df_validation)

    train_target = utils.get_target(new_train)
    validation_target = utils.get_target(new_validation)
    x_train = utils.get_x(new_train)
    x_validation = utils.get_x(new_validation)
    new_train = utils.unpop(new_train, train_target)
    new_validation = utils.unpop(new_validation, validation_target)

    classifier_result = knn(new_train, new_validation, 19)
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



