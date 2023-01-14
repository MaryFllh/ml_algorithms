import numpy as np

from activations import ReLU 
from layers import Dense
from loss import mse_loss, mse_prime


def train(X, Y, epochs, learning_rate):
    for i in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            # forward pass
            output = x
            for layer in model:
                output = layer.forward(output)
            
            error += mse_loss(output, y)
            
            # backward pass
            output_gradient = mse_prime(output, y)
            for layer in reversed(model):
               output_gradient = layer.backward(output_gradient, learning_rate)
        
        error /= len(X)
        print(f"At epoch {i + 1} mse error is {error}")




if __name__ == "__main__":
    model = [
        Dense(2, 3),
        ReLU(),
        Dense(3, 1),
        ReLU(),
    ]
    
    X = np.reshape([[0, 0], [0, 1], [1, 0], [0, 0]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    
    epochs = 1000
    learning_rate = .1
    train(X, Y, epochs, learning_rate)



