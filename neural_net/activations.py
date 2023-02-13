import numpy as np

from layers import Activation


class ReLU(Activation):
    def __init__(self):
        def relu(X):
            return np.maximum(X, 0)

        def relu_prime(X):
            return X > 0

        super().__init__(relu, relu_prime)


class Tanh(Activation):
    def __init__(self):
        def tanh(X):
            return np.tanh(X)

        def tanh_prime(X):
            return 1 - np.tanh(X) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(X):
            return 1 / (1 + np.exp(-X))

        def sigmoid_prime(X):
            return sigmoid(X) * (1 - sigmoid(X))

        super().__init__(sigmoid, sigmoid_prime)
