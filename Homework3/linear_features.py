import numpy as np
from matplotlib import pyplot as plt

def read_data():
    train_data = np.loadtxt("dataSets/lin_reg_train.txt")
    test_data = np.loadtxt("dataSets/lin_reg_test.txt")
    return train_data, test_data

class Linear_regression():
    #this class contains the features of linear regression, include linear features and polygon features
    #with fit() function, we can fit this model to the train data.
    #with pred() function, we can get the predicted label.
    def __init__(self, degree=1, lam=0):
        self.degree = degree
        self.lam = lam
    def linear_features(self, x, degree):
        Phi = np.zeros([degree+1, len(x)])
        for i in range(degree+1):
            Phi[i, :] = x ** i
        return Phi
    def fit_features(self, x, y, lam, degree):
        Phi = self.linear_features(x, degree)
        I_e = np.eye(degree + 1)
        return np.linalg.inv(Phi.dot(Phi.T) + I_e * lam).dot(Phi).dot(y) 
    def fit(self, x, y):
        self.w = self.fit_features(x, y, self.lam, self.degree)
    def pred(self, x):
        return self.linear_features(x, self.degree).T.dot(self.w)

def main():
    train_data, test_data = read_data()
    reg = Linear_regression(1, 0.01)
    reg.fit(train_data[:, 0], train_data[:, 1])
    y_pred_train = reg.pred(train_data[:, 0])
    y_pred_test = reg.pred(test_data[:, 0])
    rmse_train = np.sqrt(((y_pred_train-train_data[:, 1])**2).mean())
    rmse_test =  np.sqrt(((y_pred_test-test_data[:, 1])**2).mean())
    print("RMSE of training data is: {}".format(rmse_train))
    print("RMSE of test data is: {}".format(rmse_test))
    plt.figure();
    plt.scatter(train_data[:, 0], train_data[:, 1], marker='.', color='black', label="training data")
    x = np.arange(-1.5, 1.5, 0.01)
    plt.plot(x, reg.pred(x), color='b', label="predicted function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()