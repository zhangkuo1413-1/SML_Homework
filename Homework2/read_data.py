import numpy as np

def read_data() -> [np.ndarray, np.ndarray]:
    #read data from files
    data1 = np.loadtxt("dataSets/densEst1.txt")
    data2 = np.loadtxt("dataSets/densEst2.txt")
    return data1, data2

