import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree


if __name__=="__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=100
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = np.sum(y_test == predictions) / len(y_test)
    print(acc)