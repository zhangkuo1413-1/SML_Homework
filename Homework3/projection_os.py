from pca import *

def projection_original(X_d, u, mean, std):
    X = X_d.dot(np.linalg.pinv(u))
    X = X * std + mean
    return X

path = "dataSets/iris.txt"
X, y = read_data(path)
X_norm = normalization(X)
lam, u = pca(X)
nrmse = np.zeros([X.shape[1], X.shape[1]])
for num_d in range(X.shape[1]):
    X_d = X_norm.dot(u[:, :num_d+1])
    X_ori = projection_original(X_d, u[:, :num_d+1], X.mean(axis=0), X.std(axis=0))
    nrmse[num_d, :] = np.sqrt(((X_ori - X) ** 2).mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print(nrmse)