import cvxopt
import numpy as np
from matplotlib import pyplot as plt

class SVM():
    #SVM class
    def poly_kernel(self, x):
        return np.array([x[0]**3, 3**0.5*x[0]**2*x[1], 3**0.5*x[0]*x[1]**2,x[1]**3])
    def fit(self, X, y, C):
        #fit training data
        n = X.shape[0]
        P = cvxopt.matrix(np.zeros([n, n]))
        k = np.zeros([n, 4])
        for i in range(n):
            k[i] = self.poly_kernel(X[i])
        P = cvxopt.matrix(k.dot(k.T)*np.outer(y, y))
        q = cvxopt.matrix(-1*np.ones(n))
        G = cvxopt.matrix(np.vstack((-1 * np.eye(n), np.eye(n))))
        h = cvxopt.matrix(np.hstack((np.zeros(n), C * np.ones(n))))
        A = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(0.0)
        result = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(result["x"]).reshape(-1,)
        self.sv1 = (alpha > 1e-5)
        self.sv = np.arange(n)[(alpha > 1e-4)]
        self.weight = ((alpha * y).reshape([-1, 1]) * k).sum(axis=0)
        self.b = (-k[self.sv, :].dot(self.weight)).mean()
    def calculate_y(self, X):
        if len(X.shape) == 1:
            X = X.reshape([1, -1])
        k = np.zeros([X.shape[0], 4])
        for i in range(X.shape[0]):
            k[i] = self.poly_kernel(X[i])
        return k.dot(self.weight) + self.b
    def pred(self, X):
        res = self.calculate_y(X)
        return np.sign(res)
#plot the results
data = np.loadtxt('dataSets\iris-pca.txt')
X = data[:, 0:2]
y = data[:, 2] - 1
n = y.shape[0]
model = SVM()
model.fit(X, y, 1)
y_pred = model.pred(X)
plt.figure()
c1 = np.arange(n)[y == 1]
c2 = np.arange(n)[y == -1]
plt.scatter(X[c1, 0], X[c1, 1], c="b")
plt.scatter(X[c2, 0], X[c2, 1], c="r")
plt.show()
plt.figure()
xgrid = np.linspace(-3, 3, 601)
ygrid = np.linspace(-3, 3, 601)
xx, yy = np.meshgrid(xgrid, ygrid)
zc = model.calculate_y(np.c_[xx.ravel(), yy.ravel()])
zc = zc.reshape(xx.shape)
plt.contour(xx, yy, zc, [0.0], colors='g', linewidths = 1)
plt.contour(xx, yy, zc, [1.0], colors='g', linewidths = 0.5, linestyles='dashed')
plt.contour(xx, yy, zc, [-1.0], colors='g', linewidths = 0.5, linestyles='dashed')
c1_t = np.arange(n)[(y_pred == 1)]
c2_t = np.arange(n)[(y_pred == -1)]
c1_sv = np.arange(n)[(y_pred == 1)&(model.sv1)]
c2_sv = np.arange(n)[(y_pred == -1)&(model.sv1)]
fc = np.arange(n)[(y_pred != y)]
plt.scatter(X[c1_t, 0], X[c1_t, 1], c='y', label="C1")
plt.scatter(X[c2_t, 0], X[c2_t, 1], c='c', label="C2")
plt.scatter(X[c1_sv, 0], X[c1_sv, 1], c='y', edgecolors='g', label="support vectors")
plt.scatter(X[c2_sv, 0], X[c2_sv, 1], c='c', edgecolors='g')
plt.scatter(X[fc, 0], X[fc, 1], c='r', marker="x", s=3, alpha=0.7, label="misclassified samples")
plt.legend()
plt.show()