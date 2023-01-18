import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets


class LogisticRegression:
    def __init__(self, lr=.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            linear_preds = np.dot(X, self.weights) + self.bias
            preds = self.sigmoid(linear_preds)

            dw = (1 / n_samples) * np.dot(X.T, (preds - y))
            db = (1 / n_samples) * np.sum(preds - y) 

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X, thresh=.5):
        linear_preds = np.dot(X, self.weights) + self.bias
        preds = [1 if self.sigmoid(pred) > thresh else 0 for pred in linear_preds]
        return preds
    
    def accuracy_score(self, y, preds):
        return sum(y == preds) / len(y)

def run():
    """
    Creates a dataset, splits into train and test, fits LR and tests it.
    The mean square error is calculated and printed, the scatter plot of the
    test data points along with the fitted line are also plotted.
    """
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    lr = LogisticRegression(lr=.001)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    print(lr.accuracy_score(preds, y_test))
    


if __name__ == "__main__":
    run()
