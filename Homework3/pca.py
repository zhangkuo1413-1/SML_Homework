from data_norm import *

def pca(X):
    #impliment of PCA
    X_norm = normalization(X)
    covM = np.cov(X_norm, rowvar = False)
    return np.linalg.eig(covM)

def main():
    path = "dataSets/iris.txt"
    X, y = read_data(path)
    lam, u = pca(X)
    cum = lam.cumsum() / lam.cumsum()[-1]
    i = 1
    while cum[i - 1] < 0.95: i += 1
    print(i)
    plt.figure()
    plt.plot(np.arange(1, 5), cum)
    plt.xlabel("number of components")
    plt.ylabel("percentage of the cumulative variance")
    plt.show()

if __name__ == "__main__":
    main()