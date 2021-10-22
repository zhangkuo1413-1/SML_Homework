from linear_features import *

def main():
    train_data, test_data = read_data()
    degrees = [2, 3, 4]
    for degree in degrees:
        #we use the linear model class that was defined in the first task.
        reg_model = Linear_regression(degree, 0.01)
        reg_model.fit(train_data[:, 0], train_data[:, 1])
        y_pred_train = reg_model.pred(train_data[:, 0])
        y_pred_test = reg_model.pred(test_data[:, 0])
        rmse_train = np.sqrt(((y_pred_train-train_data[:, 1])**2).mean())
        rmse_test =  np.sqrt(((y_pred_test-test_data[:, 1])**2).mean())
        print("RMSE for degree {} of training data is: {}".format(degree, rmse_train))
        print("RMSE for degree {} of test data is: {}".format(degree, rmse_test))
        plt.figure()
        plt.scatter(train_data[:, 0], train_data[:, 1], marker='.', color='black', label="training data")
        x = np.arange(-1.5, 1.5, 0.01)
        plt.plot(x, reg_model.pred(x), color='b', label="predicted function")
        plt.title("degree of {}".format(degree))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()

