import numpy as np
from matplotlib import pyplot as plt

def gaussian_fun(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    #output the result of gaussian function by giving data and parameters
    n, d = x.shape
    a = 1 / (2 * np.pi) ** (d / 2) / np.sqrt(np.linalg.det(sigma))
    b = np.einsum("ij, jk, ik -> i", x - mu, np.linalg.inv(sigma), x - mu)
    return a * np.exp(-0.5 * b)

def plot_densities(data: dict, mu: dict, sigma: dict):
    #function to plot the data points and the probability densities of each class
    xgrid = np.linspace(-10, 10, 201)
    ygrid = np.linspace(-10, 10, 201)
    xx, yy = np.meshgrid(xgrid, ygrid)
    for dataclass, x in data.items():
        plt.scatter(x[:, 0], x[:, 1], marker='*', alpha=0.5, s=8, label='data points of {}'.format(dataclass))
        z = gaussian_fun(np.c_[xx.ravel(), yy.ravel()], mu[dataclass], sigma[dataclass])
        z = z.reshape(xx.shape)
        plt.contour(xx, yy, z, 10)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("data points and the probability densities")
    plt.show()
    
