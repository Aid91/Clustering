import numpy as np
import random

class GMM:
    def __init__(self, X, M):
        self.X = X
        self.M = M

        X_min = np.min(X[:, 0])
        X_max = np.max(X[:, 0])

        Y_min = np.min(X[:, 1])
        Y_max = np.max(X[:, 1])

        # initialization of the parameters needed for the Expectation Maximization algorithm
        self.alpha_0 = np.ones((M, 1))
        self.alpha_0 /= M

        self.mu_0 = []
        for i in range(self.M):
            self.mu_0.append(np.array([np.random.uniform(X_min, X_max), np.random.uniform(Y_min, Y_max)]))

        # initialize all covariances from covariance of the data
        cov_of_data = np.cov(X.T)
        self.sigma_0 = []
        for t in range(self.M):
            self.sigma_0.append(cov_of_data)


        # initialization of the parameters needed for the K-means algorithm
        self.mu_kmeans_0 = random.sample(list(X), M)


    def EM(max_iter, tol, interactive=False, diagonal=False):

        pass

    def k_means(max_iter, tol, interactive=False):
        pass

    def sample(N):
        pass
