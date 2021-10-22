import numpy as np
from matplotlib import pyplot as plt

def k_nearest_neighbors(x, train_data, k):
    n = len(train_data)
    if type(x) == np.ndarray:
        p = []
        for x_n in x:
            dist = abs(train_data - x_n)
            v = 2 * sorted(dist)[k-1]
            p.append(k/v/n)
        return np.array(p)
    else:
        dist = abs(train_data - x)
        v = 2 * sorted(dist)[k-1]
        p = k / v / n
        return p

def log_likelihood(test_data, train_data, k):
    p = k_nearest_neighbors(test_data, train_data, k)
    log_p = np.log(p)
    return log_p.sum()

def main():
    train_data = np.loadtxt("dataSets/nonParamTrain.txt")
    x = np.arange(-4, 8, 0.01)
    p1 = k_nearest_neighbors(x,train_data, 2)
    p2 = k_nearest_neighbors(x,train_data, 8)
    p3 = k_nearest_neighbors(x,train_data, 35)
    plt.plot(x, p1, label="K=2", linewidth = 1, color='y')
    plt.plot(x, p2, label="K=8", linewidth = 1, color='r')
    plt.plot(x, p3, label="K=35", linewidth = 1, color='b')
    plt.xlabel("x")
    plt.ylim(0, 2)
    plt.ylabel("probability density")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()