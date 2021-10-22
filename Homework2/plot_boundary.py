import numpy as np
from matplotlib import pyplot as plt
from gaussian_distribution import *

def plot_boundary(data: dict, mu: dict, sigma: dict, prior: dict):
    #function to show the probability that a point is from C1 and its boundary
    xgrid = np.linspace(-10, 10, 201)
    ygrid = np.linspace(-10, 10, 201)
    xx, yy = np.meshgrid(xgrid, ygrid)
    z1 = gaussian_fun(np.c_[xx.ravel(), yy.ravel()], mu['C1'], sigma['C1'])
    z2 = gaussian_fun(np.c_[xx.ravel(), yy.ravel()], mu['C2'], sigma['C2'])
    zc = z1 * prior['C1'] / (z1 * prior['C1'] + z2 * prior['C2'])
    zc = zc.reshape(xx.shape)
    plt.pcolormesh(xx, yy, zc, cmap=plt.cm.Blues)
    cs = plt.contour(xx, yy, zc, 10, colors='r', linewidths = 1)
    plt.clabel(cs, inline=1, fontsize=8)
    for dataclass, x in data.items():
        plt.scatter(x[:, 0], x[:, 1], marker='*', alpha=0.5, s=8, label='data points of {}'.format(dataclass))
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("The probability that a point from C1")
    plt.show()