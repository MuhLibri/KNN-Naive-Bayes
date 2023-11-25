import sys
import knn
import naive_bayes
import utils
import file
from sklearn.metrics import accuracy_score
import implement


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


df_train = file.read_csv("data_train.csv")
df_validation = file.read_csv("data_validation.csv")

new_train = utils.convert_target_to_int(df_train)
new_validation = utils.convert_target_to_int(df_validation)

train_target = utils.get_target(new_train)
validation_target = utils.get_target(new_validation)
x_train = utils.get_x(new_train)
x_validation = utils.get_x(new_validation)
new_train = utils.unpop(new_train, train_target)
new_validation = utils.unpop(new_validation, validation_target)


# new_train = convert_target_to_int(normalize(df_train, meanstd(df_train)))
# new_validation = convert_target_to_int(normalize(df_validation, meanstd(df_validation)))

# For manual implementation, k = 15 will be used
classifier_result = knn.knn(new_train, new_validation, 15)
print("Manual accuracy:", accuracy_score(validation_target, classifier_result))

file.write_csv(classifier_result, "knn_result.csv")


# For sci-kit implementation, k = 19 will be used
implement.determine_k(x_train, x_validation, train_target, validation_target)
print("Scikit-learn accuracy: ", implement.knn_sklearn(x_train, x_validation, train_target, validation_target, 19))




