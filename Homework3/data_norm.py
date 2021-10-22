import numpy as np
from matplotlib import pyplot as plt

def read_data(path):
    #function to read data
    features = np.loadtxt(path, delimiter=',')
    return features[:, :-1], features[:, -1]

def normalization(data):
    #function to normalization data
    return (data - data.mean(axis=0)) / data.std(axis=0)

path = "dataSets/iris.txt"
X, y = read_data(path)
X_norm = normalization(X)