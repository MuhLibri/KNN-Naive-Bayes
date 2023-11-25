import csv
import numpy as np
import pandas as pd


def read_csv(file):
    result = list()
    with open(file) as csvfile:
        # Read csv file
        reader = csv.reader(csvfile)
        # Ignore header
        next(reader, None)
        for row in reader:
            result.append(row)

    # Convert all elements into float
    new_result = [[float(x) for x in y] for y in result]

    # Convert all target values into integer

    return new_result


def write_csv(array, path):
    csv_array = list()
    for i in range(len(array)):
        temp = [i + 1, array[i]]
        csv_array.append(temp)

    csv_array.sort(key=lambda tup: tup[1])
    csv_array_2 = np.array(csv_array)
    pd.DataFrame(csv_array_2).to_csv('output/' + path, index=False, header=['id', 'price_range'])