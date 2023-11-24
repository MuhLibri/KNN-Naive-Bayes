import sys
import knn
import naive_bayes
import csv


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


df_train = read_csv(sys.argv[1])
df_validation = read_csv(sys.argv[2])

if sys.argv[0] == "knn.py":
    # Neighbors: 5
    knn.knn(df_train, df_validation, 5)
elif sys.argv[0] == "naive_bayes.py":
    naive_bayes.naive_bayes(df_train, df_validation)
else:
    print("Perintah tidak valid. Silakan coba lagi.")



# df_train = read_csv("data_train.csv")
# df_validation = read_csv("data_validation.csv")
#
# print(df_train)