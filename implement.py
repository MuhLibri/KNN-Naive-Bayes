from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def knn_sklearn(x_train, x_validation, train_target, validation_target, k):
    # Find the accuracy of kNN algorithm
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, train_target)

    y_pred = knn.predict(x_validation)

    print("Sci-kit accuracy:", accuracy_score(validation_target, y_pred))
    print("Sci-kit precision: ", precision_score(validation_target, y_pred, average='micro'))
    print("Sci-kit recall: ", recall_score(validation_target, y_pred, average='micro'))


def determine_k(x_train, x_validation, train_target, validation_target):
    # Find the most fitting k-value
    accuracy = list()
    for i in range(1, 40):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(x_train, train_target)
        yhat = neigh.predict(x_validation)
        accuracy.append(accuracy_score(validation_target, yhat))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), accuracy, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=10)
    plt.title('accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    print("Maximum accuracy: ", max(accuracy), "at K =", accuracy.index(max(accuracy)) + 1)
    plt.show()
