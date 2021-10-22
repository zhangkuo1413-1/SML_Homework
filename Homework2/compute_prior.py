import numpy as np

def compute_prior_fun(data1: np.ndarray, data2: np.ndarray) -> [float, float]:
    #compute the prior probability of each class from the dataset
    #the shape of input data is (n_1, 2) and (n_2, 2)
    n_1 = data1.shape[0]
    n_2 = data2.shape[0]
    return n_1 / (n_1 + n_2), n_2 / (n_1 + n_2)