import pandas as pd
from math import sqrt

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

        # Sort the distances list in descending order
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



