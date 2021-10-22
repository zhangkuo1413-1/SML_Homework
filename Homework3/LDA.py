import numpy as np
from matplotlib import pyplot as plt

def calculate_sw(X):
    #calculate the matrix S_w
    S_w = 0
    for Xi in X:
        S_w += np.cov(Xi.T)
    return S_w

def calculate_sb(X):
    #calculate the matrix S_b
    S_b = 0
    mu_all = 0
    mu = []
    n = 0
    for Xi in X:
        mu_all += Xi.sum(axis=0)
        n += Xi.shape[0]
        mu.append(Xi.mean(axis=0))
    mu_all = mu_all / n
    for Xi, mui in zip(X, mu):
        S_b += Xi.shape[0] * np.outer(mui - mu_all, mui - mu_all)
    return S_b

def calculate_w(S_w, S_b):
    #calculate the eigenvector
    lam, W = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
    return lam, W
    
def classification(data, W, X):
    #classify the data
    mu_LDA = []
    var_LDA = []
    X_LDA = []
    prior = []
    data_LDA = data.dot(W)
    discrimator = np.zeros([data_LDA.shape[0], len(X)])
    for Xi in X:
        Xi_LDA = Xi.dot(W)
        X_LDA.append(Xi_LDA)
        mu_LDA.append(Xi_LDA.mean(axis=0))
        var_LDA.append(np.cov(Xi_LDA.T))
        prior.append(Xi.shape[0] / data.shape[0])
    for i in range(len(X)):
        for j in range(data_LDA.shape[0]):
            discrimator[j, i] = -0.5 * np.linalg.det(var_LDA[i]) - 0.5 * (data_LDA[j] - mu_LDA[i]).dot(np.linalg.inv(var_LDA[i])).dot(data_LDA[j] - mu_LDA[i]) + np.log(prior[i])
    return discrimator.argmax(axis=1) + 1



data = np.loadtxt('dataSets/ldaData.txt')
y = np.zeros(137)
y[: 50] = 1
y[50 : 93] = 2
y[93 : 137] = 3
X1 = data[np.argwhere(y==1).reshape(-1)]
X2 = data[np.argwhere(y==2).reshape(-1)]
X3 = data[np.argwhere(y==3).reshape(-1)]

X = [X1, X2, X3]
S_w = calculate_sw(X)
S_b = calculate_sb(X)
lam, W = calculate_w(S_w, S_b)
y_pred = classification(data, W, X)
num_miscls = 137 - (y == y_pred).sum()
print("Number of misclassified samples: {}".format(num_miscls))
X1_pred = data[np.argwhere(y_pred==1).reshape(-1)]
X2_pred = data[np.argwhere(y_pred==2).reshape(-1)]
X3_pred = data[np.argwhere(y_pred==3).reshape(-1)]
#plot the original data
plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], color='b', label="$C_1$")
plt.scatter(X2[:, 0], X2[:, 1], color='r', label="$C_2$")
plt.scatter(X3[:, 0], X3[:, 1], color='c', label="$C_3$")
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.title("Original dataset")
plt.legend()
plt.show()
#plot the classified data
plt.figure()
plt.scatter(X1_pred[:, 0], X1_pred[:, 1], color='b', label="$C_1$")
plt.scatter(X2_pred[:, 0], X2_pred[:, 1], color='r', label="$C_2$")
plt.scatter(X3_pred[:, 0], X3_pred[:, 1], color='c', label="$C_3$")
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.title("Classified dataset")
plt.legend()
plt.show()