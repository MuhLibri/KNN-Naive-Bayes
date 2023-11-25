import csv
from argparse import ArgumentParser
import knn, naive_bayes

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
    for row in new_result:
        row[len(new_result[0])-1] = int(row[len(new_result[0])-1]) # Co
    return new_result

parser = ArgumentParser()
parser.add_argument('algorithm')
parser.add_argument('path_to_train_data')
parser.add_argument('path_to_validation_data')

if __name__ == '__main__':
    args = parser.parse_args()
    algorithm = args.algorithm
    df_train, df_validation = args.path_to_train_data, args.path_to_validation_data

    if algorithm == "knn":
        # Neighbors: 5
        knn.knn(df_train, df_validation, 5)
    elif algorithm == "naive_bayes":
        naive_bayes.naive_bayes(df_train, df_validation)
    else:
        print("Perintah tidak valid. Silakan coba lagi.")



# df_train = read_csv("data_train.csv")
# df_validation = read_csv("data_validation.csv")
#
# print(df_train)