from pca import *
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["r", "c", "b", "g"])
path = "dataSets/iris.txt"
X, y = read_data(path)
X_norm = normalization(X)
lam, u = pca(X)
cum = lam.cumsum() / lam.cumsum()[-1]
i = 1
while cum[i - 1] < 0.95: i += 1
X_d = X_norm.dot(u[:, :i])
plt.figure()
plt.scatter(X_d[:, 0], X_d[:, 1], c=y, cmap=cmap)
plt.show()