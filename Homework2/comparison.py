import numpy as np
from gaussian_kernel import log_likelihood as l_gaussian
from knn import log_likelihood as l_knn

data_test = np.loadtxt("dataSets/nonParamTest.txt")
data_train = np.loadtxt("dataSets/nonParamTrain.txt")
print("Training sets:")
print("KDE with sigma = 0.03: ", l_gaussian(data_train, data_train, 0.03))
print("KDE with sigma = 0.2: ", l_gaussian(data_train, data_train, 0.2))
print("KDE with sigma = 0.8: ", l_gaussian(data_train, data_train, 0.8))
print("KNN with K = 2: ", l_knn(data_train, data_train, 2))
print("KNN with K = 8: ", l_knn(data_train, data_train, 8))
print("KNN with K = 35: ", l_knn(data_train, data_train, 35))
print("Testing sets:")
print("KDE with sigma = 0.03: ", l_gaussian(data_test, data_train, 0.03))
print("KDE with sigma = 0.2: ", l_gaussian(data_test, data_train, 0.2))
print("KDE with sigma = 0.8: ", l_gaussian(data_test, data_train, 0.8))
print("KNN with K = 2: ", l_knn(data_test, data_train, 2))
print("KNN with K = 8: ", l_knn(data_test, data_train, 8))
print("KNN with K = 35: ", l_knn(data_test, data_train, 35))