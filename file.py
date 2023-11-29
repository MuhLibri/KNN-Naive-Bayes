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

def read_csv_pandas(file):
    return pd.read_csv(file)


def write_csv(array, headers: list[str], path: str):

    csv_array_2 = np.array(array)
    pd.DataFrame(csv_array_2).to_csv('output/' + path, index=False, header=headers)

def write_prediction_result(array: list[list[int, int]], path: str):
    write_csv(array, ['id', 'price_range'], path)

def read_bytes(file_path: str) -> bytes:
    with open(file_path, mode='rb') as fi:
        return fi.read()

def write_bytes(content: bytes, file_path: str) -> None:
    with open(file_path, mode='wb') as fi:
        fi.write(content)
