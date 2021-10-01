import math

import numpy as np


def gaussian_rbf(x: np.ndarray, xx: np.ndarray):
    sigma_sqr = xx.var()
    gamma = 1 / (2 * sigma_sqr)
    x_norm = np.sum(x**2, axis=-1)
    xx_norm = np.sum(xx**2, axis=-1)
    return np.exp(-gamma * (x_norm[:, None] + xx_norm[None, :] - 2 * np.dot(x, xx.T)))


def gaussian(x: np.ndarray, xx: np.ndarray):
    sigma_sqr = xx.var()
    x_norm = np.sum(x**2, axis=-1)
    xx_norm = np.sum(xx**2, axis=-1)
    return np.exp(-(1 / (2 * sigma_sqr)) * (x_norm[:, None] + xx_norm[None, :] - 2 * np.dot(x, xx.T)))


def laplace_rbf(x: np.ndarray, xx: np.ndarray):
    sigma = math.sqrt(xx.var())
    x_norm = np.sum(x**2, axis=-1)
    xx_norm = np.sum(xx**2, axis=-1)
    return np.exp(-(1 / sigma) * (x_norm[:, None] + xx_norm[None, :] - 2 * np.dot(x, xx.T)))


def linear(x: np.ndarray, xx: np.ndarray):
    return x
