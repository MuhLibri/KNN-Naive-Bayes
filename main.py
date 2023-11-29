import sys
import knn
import naive_bayes
import utils
import file
import implement


algorithm_option = input("Pilih algoritma yang ingin dipilih:"
                         "\n1. KNN"
                         "\n2. Naive-Bayes"
                         "\n")

if (algorithm_option == "1"):
    df_train = file.read_csv("data_train.csv")
    df_validation = file.read_csv("data_validation.csv")

    # df_train_pandas = file.read_csv_pandas("data_train.csv")
    # utils.corr_train(df_train_pandas)

    classifier_result = knn.train_knn(df_train, df_validation)

    csv_array = list()
    for i in range(len(classifier_result)):
        temp = [i + 1, classifier_result[i]]
        csv_array.append(temp)

    file.write_prediction_result(csv_array, "knn_result.csv")

    print('Predicting classification for test data...')
    df_test = file.read_csv("test.csv")
    test_result = knn.predict_knn(df_train, df_test)

    test_array = list()
    for i in range(len(test_result)):
        temp = [i, test_result[i]]
        test_array.append(temp)

    print('Writing classification result to output/knn_test_result_2.csv')
    file.write_prediction_result(test_array, "knn_test_result_2.csv")


elif (algorithm_option == "2"):
    df_train, df_valid = file.read_csv('data_train.csv'), file.read_csv('data_validation.csv')
    prediction = naive_bayes.predict(df_train, df_valid)

else:
    print("Invalid input. Try again")








