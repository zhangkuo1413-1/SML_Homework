import numpy as np

def gaussian_biased_estimate(data: np.ndarray) -> [float, float]:
    #give a array with shape (n, 2), then estimate the mean and covariance of the data in the array
    n = data.shape[0]
    mu = data.sum(axis=0) / n
    sigma = (data - mu).T.dot(data - mu) / n
    return mu, sigma

def gaussian_unbiased_estimate(data: np.ndarray) -> [np.ndarray, np.ndarray]:
    #give a array with shape (n, 2), then estimate the mean and covariance of the data in the array
    #outputs are mean with shape (2, ) and covariance with shape (2, 2)
    n = data.shape[0]
    mu = data.sum(axis=0) / n
    sigma = (data - mu).T.dot(data - mu) / (n - 1)
    return mu, sigma