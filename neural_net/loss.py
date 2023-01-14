import numpy as np

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def mse_prime(pred, target):
    return 2 * (pred - target) / np.size(target)