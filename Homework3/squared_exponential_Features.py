from bayesian_regression import *

class SEFeatures(BayesianRegression):
    #This class is the model of SE-features. We also use some functions in Bayesian model.
    def __init__(self, degree=20, lam=0.01, sigma=0.1):
        super().__init__(degree=degree, lam=lam, sigma=sigma)
    
    def se_feature(self, x):
        Phi = np.zeros([self.degree+2, len(x)])
        Phi[0, :] = np.zeros(len(x))
        for i in range(self.degree):
            Phi[i+1, :] = np.exp(-0.5 * 10 * (x - i * 0.1 + 1) ** 2)
        return Phi

    def fit(self, x, y):
        self.Phi = self.se_feature(x)
        self.y = y

    def pred(self, x):
        I_e = np.eye(self.degree + 2)
        w = np.linalg.inv(self.lam * self.sigma_qua * I_e + self.Phi.dot(self.Phi.T)).dot(self.Phi).dot(self.y)
        phi_x = self.se_feature(x)
        mu = phi_x.T.dot(w)
        intermedia = np.linalg.inv(self.lam * I_e + self.Phi.dot(self.Phi.T) / self.sigma_qua)
        sigma_qua = self.sigma_qua + np.einsum("ji, jk, ki -> i", phi_x, intermedia, phi_x)
        return mu, sigma_qua

def main():
    train_data, test_data = read_data()
    reg = SEFeatures()
    reg.fit(train_data[:, 0], train_data[:, 1])
    mu_train, var_train = reg.pred(train_data[:, 0])
    mu_test, var_test = reg.pred(test_data[:, 0])
    rmse_train = np.sqrt(((mu_train-train_data[:, 1])**2).mean())
    rmse_test =  np.sqrt(((mu_test-test_data[:, 1])**2).mean())
    print("RMSE of training data is: {}".format(rmse_train))
    print("RMSE of test data is: {}".format(rmse_test))
    allh_train = reg.cal_likelihood(train_data[:, 0], train_data[:, 1])
    allh_test = reg.cal_likelihood(test_data[:, 0], test_data[:, 1])
    print("average log-likelihood of training data is: {}".format(allh_train))
    print("average log-likelihood of test data is: {}".format(allh_test))
    x = np.arange(-1.5, 1.5, 0.01)
    mu, var = reg.pred(x)
    std = np.sqrt(var)
    for i in [1, 2, 3]:
        plt.figure();
        plt.scatter(train_data[:, 0], train_data[:, 1], marker='.', color='black', label="training data")
        plt.plot(x, mu, color='b', label="predicted function")
        plt.fill_between(x, mu - i * std, mu + i * std, facecolor="blue", alpha=0.3)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("{} standard deviations".format(i))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()