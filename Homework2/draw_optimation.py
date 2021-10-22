import numpy as np
from matplotlib import pyplot as plt

def rosenbrock_fun(x: np.ndarray) -> float:
    #shape of input array: (n, ), n must equal or higher than 2 
    sum = 0
    for i in range(len(x)-1):
        sum += 100 * (x[i+1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return sum

def derivative_rosenbrock(x: np.ndarray) -> np.ndarray:
    #shape of input array: (n, ), shape of output array: (n, )
    n = len(x)
    grad = np.zeros(n)
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1)
    for i in range(1, n-1):
        grad[i] = 200 * (x[i] - x[i-1] ** 2) - 400 * x[i] * (x[i+1] - x[i] ** 2) + 2 * (x[i] - 1)
    grad[n-1] = 200 * (x[n-1] - x[n-2] ** 2)
    return grad

def check_grad(x: np.ndarray):
    #function to check whether the derivative is correct
    grad = np.zeros(len(x))
    fx = rosenbrock_fun(x)
    fx_d = derivative_rosenbrock(x)
    for i in range(len(x)):
        x_theta = x.copy()
        x_theta[i] += 1e-6
        grad[i] = (rosenbrock_fun(x_theta) - fx) / 1e-6
    mse = abs(fx_d - grad).sum() / len(x)
    if mse < 0.01:
        print("gradient checked!")
    else:
        print("wrong deriviation")

def optimization_process(x: np.ndarray, learning_rate: float, steps: int) -> [np.ndarray, np.ndarray]:
    value_op = np.zeros(steps + 1)
    value_op[0] = rosenbrock_fun(x)
    for i in range(steps):
        grad = derivative_rosenbrock(x)
        x -= grad * learning_rate
        value_op[i + 1] = rosenbrock_fun(x)
    return x, value_op

def main():
    n = 20
    x = (np.random.rand(n) - 0.5) * 2
    check_grad(x)
    learning_rate = 3e-6
    x, op = optimization_process(x, learning_rate, 10000)
    print(x)
    print("the value after optimization is {}".format(op[-1]))
    plt.plot(np.arange(10001), op, label="learning_rate={}".format(learning_rate))
    plt.title("Process of Optimization")
    plt.xlabel("step")
    plt.ylabel("value of Rosenbrock")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()