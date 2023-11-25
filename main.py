import sys
import knn
import naive_bayes
import csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from numpy import mean, std


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


# df_train = read_csv(sys.argv[1])
# df_validation = read_csv(sys.argv[2])
#
# if sys.argv[0] == "knn.py":
#     # Neighbors: 5
#     knn.knn(df_train, df_validation, 5)
# elif sys.argv[0] == "naive_bayes.py":
#     naive_bayes.naive_bayes(df_train, df_validation)
# else:
#     print("Perintah tidak valid. Silakan coba lagi.")


df_train = read_csv("data_train.csv")
df_validation = read_csv("data_validation.csv")

new_train = convert_target_to_int(df_train)
new_validation = convert_target_to_int(df_validation)

# new_train = convert_target_to_int(normalize(df_train, meanstd(df_train)))
# new_validation = convert_target_to_int(normalize(df_validation, meanstd(df_validation)))

classifier_result = knn.knn(new_train, new_validation, 7)
print(classifier_result)

train_target = get_target(new_train)
validation_target = get_target(new_validation)
print(validation_target)

print(accuracy_score(validation_target, classifier_result))

# Import sklearn method

x_train = get_x(new_train)
x_validation = get_x(new_validation)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, train_target)

y_pred = knn.predict(x_validation)

accuracy = accuracy_score(validation_target, y_pred)
print("Accuracy:", accuracy)

