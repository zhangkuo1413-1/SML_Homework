import numpy as np
from matplotlib import pyplot as plt

def target_function(x):
    return np.sin(x) + np.sin(x) ** 2 + np.random.normal(loc=0,scale=np.sqrt(0.005),size=(len(x))) 

def kernel(x1, x2, sigma):
    return np.exp(-((x1 - x2) ** 2) / (2 * sigma ** 2))

def calculate_k(x, xnp, sigma):
    k = np.zeros(len(x))
    for i in range(len(x)):
        k[i] = kernel(x[i], xnp, sigma)
    return k

def plot_process(i, x, X, t, m, var):
    #plot the figure
    plt.figure()
    plt.plot(x, m, c="r", label="predicted function")
    plt.plot(x, np.sin(x) + np.sin(x) ** 2, c="g", label="ground truth")
    plt.scatter(X[:-1], t[:-1], c="c", marker="o", label="sampling")
    plt.scatter(X[-1], t[-1], c="b", marker="*", label="new point")
    plt.fill_between(x, m-2*np.sqrt(var), m+2*np.sqrt(var), facecolor="r", alpha=0.2, label="standard deviation")
    plt.title("Iteration: {}".format(i+1))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

#Here is the training process
x = np.arange(0, 2*np.pi+0.005, 0.005)
x[-1] = 2 * np.pi
y = target_function(x)
X = [x[0]]
t = [y[0]]
C = np.array([[kernel(x[0], x[0], 1) + 0.005]])
m = np.zeros(len(x))
var = np.zeros(len(x))
i_temp = 0
plot_num=[1, 2, 5, 10, 15]
for j in range(15):
    for i in range(len(x)):
        k_temp = calculate_k(np.array(X), x[i], 1)
        c_temp = kernel(x[i], x[i], 1) + 0.005
        m[i] = k_temp.dot(np.linalg.inv(C)).dot(np.array(t))
        var[i] = c_temp - k_temp.dot(np.linalg.inv(C)).dot(k_temp)
    if j + 1 in plot_num:
       plot_process(j, x, X, t, m, var)
    i_temp = var.argmax(axis=0)
    k = calculate_k(np.array(X), x[i_temp], 1)
    c = kernel(x[i_temp], x[i_temp], 1) + 0.005
    C = np.vstack((np.hstack((C, k.reshape([-1, 1]))), np.hstack((k.reshape([1, -1]), np.array([[c]])))))
    X.append(x[i_temp])
    t.append(y[i_temp])
