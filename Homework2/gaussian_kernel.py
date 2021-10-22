import numpy as np
from matplotlib import pyplot as plt

def gaussian_fun(x, width):
    f1 = 1 / np.sqrt(2 * np.pi * width ** 2)
    f2 = np.exp(- x ** 2 / (2 * width ** 2))
    return f1 * f2

def gaussian_kernel(x, train_data, width):
    if type(x) == np.ndarray:
        p = np.zeros(x.shape)
    else:
        p = 0
    for x_n in train_data:
        p += gaussian_fun(x - x_n, width)
    return p / len(train_data)

def log_likelihood(test_data, train_data, width):
    p = gaussian_kernel(test_data, train_data, width)
    log_p = np.log(p)
    return log_p.sum()

def main():
    train_data = np.loadtxt("dataSets/nonParamTrain.txt")
    x = np.arange(-4, 8, 0.005)
    p1 = gaussian_kernel(x, train_data, 0.03)
    p2 = gaussian_kernel(x, train_data, 0.2)
    p3 = gaussian_kernel(x, train_data, 0.8)
    print("The log-likelihood with sigma=0.03:", log_likelihood(train_data, train_data, 0.03))
    print("The log-likelihood with sigma=0.2:", log_likelihood(train_data, train_data, 0.2))
    print("The log-likelihood with sigma=0.8:", log_likelihood(train_data, train_data, 0.8))
    plt.plot(x, p1, label="sigma = 0.03")
    plt.plot(x, p2, label="sigma = 0.2")
    plt.plot(x, p3, label="sigma = 0.8")
    plt.xlabel("x")
    plt.ylabel("probability density")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
