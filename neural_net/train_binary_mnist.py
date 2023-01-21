import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms

from activations import Sigmoid
from loss import binary_ce_loss, binary_ce_loss_prime
from layers import Convolutional, Dense, Reshape


def load_data(train=True):
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST('../data', train=train, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset)

    return train_loader

def preprocess_data(data_loader):
    """
    Extracts only images with labels 0, 1 for binary classification
    """
    x, y = [], []
    for _, (data, label) in enumerate(data_loader):
        if label in [0, 1]:
            x.extend(data.numpy())
            one_hot_label = nn.functional.one_hot(label, num_classes=2)
            y.extend(one_hot_label.numpy())
    x, y = np.asarray(x), np.asarray(y)
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float") / 255
    y = y.reshape(len(y), 2, 1)
    return x, y

def train(X_train, y_train, model, loss_function, loss_function_prime, epoch=1000, learning_rate=.01):
    for i in range(epoch):
        error = 0
        for X, y in zip(X_train, y_train):
            
            # forward
            pred = predict(X, model)
            error += loss_function(pred, y)
            
            # backward
            output_gradient = loss_function_prime(pred, y)
            for layer in reversed(model):
                output_gradient = layer.backward(output_gradient, learning_rate)
        error /= len(X)
        print(f"At epoch {i} the error is {error}")
    
def predict(X_test, model):
    output = X_test
    for layers in model:
        output = layers.forward(output)
    return output


if __name__=="__main__":
    train_loader = load_data()
    test_loader = load_data(train=False)
    X_train, y_train = preprocess_data(train_loader)[:100]
    X_test, y_test = preprocess_data(test_loader)[:100]

    model = [
        Convolutional((1, 28, 28), (5, 3, 3)),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]
    
    train(
        X_train, 
        y_train,
        model,
        binary_ce_loss,
        binary_ce_loss_prime
    )

    for x, y in zip(X_test, y_test):
        output = predict(x, model)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")