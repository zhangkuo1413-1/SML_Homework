import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

def init_parameters():
    #Initialize the parameters
    #np.random.seed(123)
    mu = np.array([[0, 2], [-1, 0], [1, 1], [2, -1]])
    #mu = np.random.rand(4, 2)*0.5
    sigma = np.random.rand(4, 2, 2) * 2
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    return mu, sigma, pi

def e_step(data, mu, sigma, pi):
    #E-Step
    alpha = np.zeros([data.shape[0], mu.shape[0]])
    for i in range(mu.shape[0]):
        alpha[:, i] = pi[i] * multivariate_normal.pdf(data, mean=mu[i], cov=np.diag(sigma[i]))
    alpha = alpha / alpha.sum(axis=1).reshape(500, -1)
    return alpha

def m_step(data, alpha):
    #M-step
    mu = np.zeros([alpha.shape[1], data.shape[1]])
    sigma = np.zeros([alpha.shape[1], data.shape[1], data.shape[1]])
    pi = np.zeros(alpha.shape[1])
    N = alpha.sum(axis=0)
    for i in range(alpha.shape[1]):
        mu[i] = alpha[:, i].dot(data) / N[i]
        sigma[i] = (alpha[:, i].reshape([-1, 1]) * (data - mu[i])).T.dot(data - mu[i]) / N[i]
        pi[i] = N[i] / data.shape[0]
    return mu, sigma, pi

def likelihood_fun(data, mu, sigma, pi):
    #to calculate the likelihood of data under this distribution
    alpha = np.zeros([data.shape[0], mu.shape[0]])
    for i in range(len(pi)):
        alpha[:, i] = pi[i] * multivariate_normal.pdf(data, mean=mu[i], cov=sigma[i])
    p = alpha.sum(axis=1)
    return np.log(p).sum()

def iteration_process(data, number):
    #Iteration of E-Step and M-Step
    mu, sigma, pi = init_parameters()
    likelihood = np.zeros(number)
    for i in range(number):
        alpha = e_step(data, mu, sigma, pi)
        mu, sigma, pi = m_step(data, alpha)
        likelihood[i] = likelihood_fun(data, mu, sigma, pi)
    return mu, sigma, pi, likelihood

def plot_result(alpha, mu, sigma, data, t):
    #plot the result in a figure
    label = alpha.argmax(axis=1).tolist()
    colors = ['r', 'b', 'g', 'c']
    xgrid = np.linspace(-2, 3.5, 551)
    ygrid = np.linspace(-2, 5, 701)
    xx, yy = np.meshgrid(xgrid, ygrid)
    for i in range(4):
        s = np.array([data[j] for j in range(len(label)) if label[j] == i])
        if s.size != 0:
            plt.scatter(s[:, 0], s[:, 1], marker='*', color=colors[i], label="Gaussian Model {}".format(i+1))
        z = multivariate_normal.pdf(np.c_[xx.ravel(), yy.ravel()], mean=mu[i], cov=sigma[i])
        z = z.reshape(xx.shape)
        plt.contour(xx, yy, z, 1, colors='black')
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Iteration steps: {}".format(t))
    plt.show()


data = np.loadtxt("dataSets/gmm.txt")
mu, sigma, pi, li = iteration_process(data, 30)
plt.plot(np.arange(30)+1, li, color='b')
plt.xlabel("Steps")
plt.ylabel("Log-likelihood")
plt.title("The process of EM")
plt.show()
t_n = [1, 3, 5, 10, 30]
for t in t_n:
    mu, sigma, pi, li = iteration_process(data, t)
    alpha = e_step(data, mu, sigma, pi)
    plot_result(alpha, mu, sigma, data, t)