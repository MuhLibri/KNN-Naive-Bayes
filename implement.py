from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def knn_sklearn(x_train, x_validation, train_target, validation_target):
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train, train_target)

    y_pred = knn.predict(x_validation)

    accuracy = accuracy_score(validation_target, y_pred)
    return accuracy