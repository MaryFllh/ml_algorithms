import numpy as np


def mse_loss(pred: np.array, target: np.array):
    return np.mean((pred - target) ** 2)


def mse_prime(pred: np.array, target: np.array):
    return 2 * (pred - target) / np.size(target)


def binary_ce_loss(pred: np.array, target: np.array):
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


def binary_ce_loss_prime(pred: np.array, target: np.array):
    return (((1 - target) / (1 - pred)) - target / pred) / np.size(target)
