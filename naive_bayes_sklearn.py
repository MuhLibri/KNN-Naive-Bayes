from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def nb_sklearn(x_train, x_validation, train_target, validation_target):
    # Find the accuracy of Naive-Bayes algorithm
    nb = GaussianNB()
    nb.fit(x_train, train_target)

    y_pred = nb.predict(x_validation)

    accuracy = accuracy_score(validation_target, y_pred)
    return accuracy

