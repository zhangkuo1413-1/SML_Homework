import numpy as np
from matplotlib import pyplot as plt

def plot_histogram_train(train_data: np.ndarray, size: float) -> np.ndarray:
    #give a train data and to estimate data. This function give the estimated probability.
    bins = np.arange(np.floor(train_data.min()), np.ceil(train_data.max())+size, size)
    plt.hist(train_data, bins=bins, color='brown', alpha=0.8)
    plt.title("Histogram with bin of size {}".format(size))
    plt.xlabel("x")
    plt.ylabel("number of samples")
    plt.show()

def main():
    train_data = np.loadtxt("dataSets/nonParamTrain.txt")
    test_data = np.loadtxt("dataSets/nonParamTest.txt")
    plot_histogram_train(train_data, 0.02)
    plot_histogram_train(train_data, 0.5)
    plot_histogram_train(train_data, 2)
    plot_histogram_train(test_data, 0.02)

if __name__ == "__main__":
    main()