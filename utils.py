import csv


# def minmax(dataset):
#     minmax_array = list()
#     for i in range(len(dataset[0])):
#         col_values = [row[i] for row in dataset]
#         value_min = min(col_values)
#         value_max = max(col_values)
#         minmax_array.append([value_min, value_max])
#     return minmax_array
#
# def meanstd(dataset):
#     meanstd_array = list()
#     for i in range(len(dataset[0])):
#         col_values = [row[i] for row in dataset]
#         value_mean = mean(col_values)
#         value_std = std(col_values)
#         meanstd_array.append([value_mean, value_std])
#     return meanstd_array
#
# def normalize(dataset, meanstd_array):
#     new_dataset = dataset
#     for row in new_dataset:
#         normalized_columns = [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
#         for column in normalized_columns:
#             row[column - 1] = (row[column - 1] - meanstd_array[column - 1][0]) / (meanstd_array[column - 1][1])
#     return new_dataset

def get_target(dataset):
    res = list()
    for row in dataset:
        res.append(row[len(dataset[0]) - 1])
    return res

def get_x(dataset):
    res = dataset
    for row in res:
        row.pop()
    return res


def convert_target_to_int(dataset):
    for row in dataset:
        row[len(dataset[0])-1] = int(row[len(dataset[0])-1])
    return dataset

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
