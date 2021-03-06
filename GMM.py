import numpy as np
import random
import matplotlib.pyplot as plt
import utils
import numpy.random as rd

class GMM:
    def __init__(self, X, M):
        self.X = X
        self.M = M

        self.X_min = np.min(X[:, 0])
        self.X_max = np.max(X[:, 0])

        self.Y_min = np.min(X[:, 1])
        self.Y_max = np.max(X[:, 1])

        # initialization of the parameters needed for the Expectation Maximization algorithm
        self.alpha_0 = np.ones((M, 1))
        self.alpha_0 /= M

        self.mu_0 = []
        for i in range(self.M):
            self.mu_0.append(np.array([np.random.uniform(self.X_min, self.X_max), np.random.uniform(self.Y_min, self.Y_max)]))

        # initialize all covariances from covariance of the data
        cov_of_data = np.cov(X.T)
        self.sigma_0 = []
        for t in range(self.M):
            self.sigma_0.append(cov_of_data)

        self.mu = self.mu_0
        self.sigma = self.sigma_0
        self.alpha = self.alpha_0

        # initialization of the parameters needed for the K-means algorithm
        self.mu_kmeans_0 = random.sample(list(X), M)

        self.mu_kmeans = self.mu_kmeans_0


    def EM(self, max_iter, tol, interactive=False, diagonal=False):
        N, P = self.X.shape

        log_likelihood_old = 0
        log_likelihood = []

        for k in range(max_iter):
            log_likelihood_new = 0

            if interactive:
                plt.plot(self.X[:, 0], self.X[:, 1], 'o', alpha=0.1)
                plt.ion()

            # E-step
            # calculate the r_mn matrix
            r_mn = np.zeros((self.M, N))
            for j in range(self.M):
                r_mn[j, :] = self.alpha[j] * utils.multivariate_normal(self.mu[j], self.sigma[j]).pdf(self.X)

            r_mn /= r_mn.sum(axis=0)

            if interactive:
                # do the soft-classification and plot the values
                utils.classify_and_plot_data(X=self.X, r_mn=r_mn, M=self.M)
                plt.pause(0.001)

            # M-step
            # calculate alpha, mu and Sigma
            self.alpha = r_mn.sum(axis=1) / N
            self.mu = np.dot(r_mn, self.X) / np.reshape(r_mn.sum(axis=1), (self.M, 1))

            Sigma = np.zeros((self.M, P, P))
            for j in range(self.M):
                diff = self.X - self.mu[j, :]
                Sigma[j] = (r_mn[j, :, None, None] * np.einsum('ijk,ikl->ijl', diff[:, :, None], diff[:, None, :])).sum(
                    axis=0)
                Sigma[j] /= r_mn[j, :].sum(axis=0)

                if diagonal:
                    Sigma[j] = np.diag(np.diag(Sigma[j]))

            self.sigma = Sigma

            # calculate the Log-likelihood
            for alpha_t, mu_t, sigma_t in zip(self.alpha, self.mu, self.sigma):
                log_likelihood_new += alpha_t * utils.likelihood_bivariate_normal(self.X, mu_t, sigma_t)

            log_likelihood_new = np.log(log_likelihood_new).sum()
            log_likelihood.append(log_likelihood_new)

            # stopping criteria
            if np.abs(log_likelihood_new - log_likelihood_old) < tol:
                break

            log_likelihood_old = log_likelihood_new

            if interactive:
                plt.clf()
                plt.cla()

        if not interactive:
            # do the soft-classification and plot the final result
            utils.classify_and_plot_data(X=self.X, r_mn=r_mn, M=self.M)

            plt.plot(self.X[:, 0], self.X[:, 1], 'o', alpha=0.1)
            for i in range(self.M):
                utils.plot_gauss_contour(mu=self.mu[i], cov=self.sigma[i], xmin=self.X_min, xmax=self.X_max, ymin=self.Y_min, ymax=self.Y_max,
                                   title='Final results of the EM algorithm')
            plt.show()

        return log_likelihood

    def k_means(self, max_iter, tol, interactive=False):

            N, P = self.X.shape

            distance_old = 0
            distances = []

            for i in range(max_iter):

                if interactive:
                    plt.ion()

                # E-step:
                # Calculate the r_mn matrix
                dist = np.zeros((self.M, N))

                for j in range(self.M):
                    dist[j, :] = np.linalg.norm(self.X - self.mu_kmeans[j], axis=1)

                # find the minimum distances for all M components and create one-hot vectors from obtained indexes of minimum distances
                min = np.argmin(dist, axis=0)
                r_mn = (np.eye(self.M)[min]).T

                if interactive:
                    # do the hard-classification and plot the result
                    utils.classify_and_plot_data(X=self.X, r_mn=r_mn, M=self.M)
                    plt.pause(0.1)

                # update the means
                self.mu_kmeans = np.dot(r_mn, self.X) / np.reshape(r_mn.sum(axis=1), (self.M, 1))

                # calculate the distance
                distance_new = 0
                for m in range(self.M):
                    d = self.X - self.mu_kmeans[m]
                    distance_new += (r_mn[m, :] * np.linalg.norm(d, axis=1)).sum(axis=0)

                distances.append(distance_new)

                # stopping criteria
                if np.abs(distance_old - distance_new) < tol:
                    break

                distance_old = distance_new

                if interactive:
                    plt.clf()
                    plt.cla()

            if not interactive:
                # do the hard-classification and plot the final result
                utils.classify_and_plot_data(X=self.X, r_mn=r_mn, M=self.M)

            return distances


    def sample(self, N):

        M = len(self.mu)
        Y = []

        # create the array X based on the number of components and sample from discrete pmf based on values alpha
        X = np.array(np.arange(self.M))
        X = np.random.choice(X, N, p=self.alpha)

        for i in range(M):
            Y.append(rd.multivariate_normal(mean=self.mu[i], cov=self.sigma[i], size=np.sum(X == i)))

        return Y
