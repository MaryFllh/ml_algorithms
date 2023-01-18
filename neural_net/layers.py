import numpy as np

from scipy import signal

class Dense:
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input: np.array):
        self.input = input
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient: np.array, learning_rate: float):
        dw = np.dot(output_gradient, self.input.T)
        db = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= dw * learning_rate
        self.bias -= db * learning_rate
        return input_gradient


class Convolutional:
    def __init__(self, input_shape: tuple, kernal_shape: tuple, stride: int = 1):
        self.kernal_shape = kernal_shape
        self.input_shape = input_shape
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernal_depth, self.kernal_height, self.kernal_width = kernal_shape
        self.output_height = (self.input_height - self.kernal_height + 1) / stride
        self.output_width = (self.input_width - self.kernal_width + 1) / stride
        self.output_depth = self.kernal_depth

        self.kernals = np.random.randn(*self.kernal_shape)
        self.biases = np.random.randn(self.output_depth, self.output_height, self.output_width)
    
    def forward(self, input: np.array):
        self.input = input
        output = np.copy(self.biases)
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                output += signal.correlate2d(self.input[j], self.kernals[i, j], "valid")
        return output
    
    def backward(self, output_gradient, learning_rate):
        kernal_gradient = np.zeros(*self.kernal_shape)
        input_gradient = np.zeros(*self.input_shape)

        for i in range(self.output_depth):
            for j in range(self.input_depth):
                kernal_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernals[i, j], "full")
        self.kernals -= kernal_gradient * learning_rate
        self.biases -= output_gradient * learning_rate
        return input_gradient
        

class Reshape:
    def __init__(self, input_shape: int, output_shape: int):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate=None):
        return np.reshape(output_gradient, self.input_shape)


class Activation:
    def __init__(self, activation, activation_prime):
        """
        Args:
            activation(func): the activation function
            activation_prime(func): the derivative of the activation function
        """
        self.activation = activation
        self.activation_prime = activation_prime


    def forward(self, input: np.array):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Dropout:
    def __init__(self, dropout=.1):
        self.dropout = dropout

    def forward(self, input: np.array):
        self.mask = np.random.rand(input.shape[0], input.shape[1]) < (1 - self.dropout)
        return np.multiply(input, self.mask) / (1 - self.dropout)
    
    def backward(self, output_gradient: np.array, learning_rate):
        return np.multiply(output_gradient, self.mask) / (1 - self.dropout)
