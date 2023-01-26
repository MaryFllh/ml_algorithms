import numpy as np

from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils import euclidean_distance

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        return [self.find_nearest_neighbors(x) for x in X]

    def find_nearest_neighbors(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        nearest_neighbors_idx = np.argsort(distances)[:self.k]
        nearest_neighbors = [self.Y_train[i] for i in nearest_neighbors_idx]
        return Counter(nearest_neighbors).most_common()[0][0]

def run():
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.2, random_state=1)

    knn = KNN()
    knn.fit(X_train, Y_train)
    preds = knn.predict(X_test)
    print(preds)
    accuracy = sum(preds == Y_test) / len(preds)
    print(accuracy)

if __name__ == "__main__":
    run()