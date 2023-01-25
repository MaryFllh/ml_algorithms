import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class NaiveBayes:
    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        classes_num = len(self.unique_classes)
        samples_num, features_num = X.shape

        self._mean = np.zeros((classes_num, features_num), dtype=np.float64)
        self._var = np.zeros((classes_num, features_num), dtype=np.float64)
        self._priors = np.zeros(classes_num, dtype=np.float64)

        for idx, c in enumerate(self.unique_classes):
            X_c = X[y == c] # gives features of class c
            self._mean[idx, :] = np.mean(X_c, axis=0)
            self._var[idx, :] = np.var(X_c, axis=0)
            self._priors[idx] = len(X_c) / samples_num

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        posteriors = []
        for idx, _ in enumerate(self.unique_classes):
            likelihood = self._compute_likelihood(idx, x)
            prior = np.log(self._priors[idx])
            posterior = prior + likelihood
            posteriors.append(posterior)
        return np.argmax(posteriors)
    
    def _compute_likelihood(self, idx, x):
        nominator = np.exp(-(x - self._mean[idx]) ** 2 / (2 * self._var[idx]))
        denominator = np.sqrt(2 * np.pi * self._var[idx])
        return  np.sum(np.log(nominator / denominator))


if __name__ == "__main__":

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))