import sys
import knn
import naive_bayes
import utils
import file
import implement

valid = False
while not valid:
    algorithm_option = input(
        "Select one of the algorithms below:"
        "\n1. KNN"
        "\n2. Naive-Bayes"
        "\nYour choice: "
    )

    if (algorithm_option == "1"):
        df_train = file.read_csv("data_train.csv")
        df_validation = file.read_csv("data_validation.csv")

        # df_train_pandas = file.read_csv_pandas("data_train.csv")
        # utils.corr_train(df_train_pandas)

        print('Predicting classification for validation data...')
        classifier_result = knn.train_knn(df_train, df_validation)

        csv_array = list()
        for i in range(len(classifier_result)):
            temp = [i + 1, classifier_result[i]]
            csv_array.append(temp)

        print('Writing classification result to output/knn_result.csv')
        file.write_prediction_result(csv_array, "knn_result.csv")

        print('Predicting classification for test data...')
        df_test = file.read_csv("test.csv")
        prediction = knn.predict_knn(df_train, df_test)

        test_array = list()
        for i in range(len(prediction)):
            temp = [i, prediction[i]]
            test_array.append(temp)

        print('Writing classification result to output/knn_test_result_2.csv')
        file.write_prediction_result(test_array, "knn_test_result_2.csv")

        valid = True

    elif (algorithm_option == "2"):
        file_path = input('Path to model file (or leave empty to generate new model from training data):\n')
        if file_path != '':
            print(f'Loading model definition from {file_path}...')
            model = naive_bayes.read_model_from_file(file_path)
        else:
            print('Predicting classification for validation data...')
            df_train, df_valid = file.read_csv('data_train.csv'), file.read_csv('data_validation.csv')
            model, prediction = naive_bayes.predict(df_train, df_valid)

            print('Writing classification result to output/nb_result.csv')
            csv_array = list()
            for i in range(len(prediction)):
                temp = [int(i + 1), int(prediction[i])]
                csv_array.append(temp)
            file.write_prediction_result(csv_array, "nb_result.csv")

        print('Predicting classification for test data...')
        df_test = file.read_csv('test.csv')
        prediction = naive_bayes.do_predict(utils.exclude_id(df_test), model)

        test_array = list()
        for i in range(len(prediction)):
            temp = [int(i), int(prediction[i])]
            test_array.append(temp)

        print('Writing classification result to output/nb_test_result_2.csv')
        file.write_prediction_result(test_array, "nb_test_result_2.csv")

        valid = True

    else:
        print("Invalid input! Try again.")
