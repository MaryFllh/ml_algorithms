import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        dw = np.dot(output_gradient, self.input.T)
        db = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= dw * learning_rate
        self.bias -= db * learning_rate
        return input_gradient

class Activation:
    def __init__(self, activation, activation_prime):
        """
        Args:
            activation(func): the activation function
            activation_prime(func): the derivative of the activation function
        """
        self.activation = activation
        self.activation_prime = activation_prime


    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


