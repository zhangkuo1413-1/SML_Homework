import numpy as np
from matplotlib import pyplot as plt
from compute_prior import *
from read_data import *
from parameter_estimate import *
from gaussian_distribution import *
from plot_boundary import *

def main():
    data_den1, data_den2 = read_data()
    prior1, prior2 = compute_prior_fun(data_den1, data_den2)
    prior = {'C1': prior1, 'C2': prior2}
    print("the prior of C1 is {}".format(prior1))
    print("the prior of C2 is {}".format(prior2))
    mu1_biased, sigma1_biased = gaussian_biased_estimate(data_den1)
    mu1_unbiased, sigma1_unbiased = gaussian_unbiased_estimate(data_den1)
    mu2_biased, sigma2_biased = gaussian_biased_estimate(data_den2)
    mu2_unbiased, sigma2_unbiased = gaussian_unbiased_estimate(data_den2)
    print("the mean of C1 is:\n", mu1_biased)
    print("the covariance of C1 is:\nbiased:\n", sigma1_biased, "\nunbiased:\n", sigma1_unbiased)
    print("the mean of C2 is:\n", mu2_biased)
    print("the covariance of C2 is:\nbiased:\n", sigma2_biased, "\nunbiased:\n", sigma2_unbiased)
    data = {'C1': data_den1, 'C2': data_den2}
    mu = {'C1': mu1_unbiased, 'C2': mu2_unbiased}
    sigma = {'C1': sigma1_unbiased, 'C2': sigma2_unbiased}
    plot_densities(data, mu, sigma)
    plot_boundary(data, mu, sigma, prior)

if __name__ == "__main__":
    main()
