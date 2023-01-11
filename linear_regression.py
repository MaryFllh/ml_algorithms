import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression:
    def __init__(self, lr=.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            preds = np.dot(X, self.weights) + self.bias
            dw = (2 / n_samples ) * np.dot(X.T, (preds - y))
            db = (2 / n_samples) * sum(preds - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db 
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def run():
    """
    Creates a dataset, splits into train and test, fits LR and tests it.
    The mean square error is calculated and printed, the scatter plot of the
    test data points along with the fitted line are also plotted.
    """
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=10, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    
    lr = LinearRegression(lr=.01)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    
    mse = np.mean((preds - y_test) ** 2)
    print(mse)

    plt.scatter(X_test, y_test, s=10)
    plt.plot(X_test, preds)
    plt.show()


if __name__ == "__main__":
    run()
