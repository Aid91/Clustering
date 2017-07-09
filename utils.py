from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def likelihood_bivariate_normal(X, mu, cov):
    """Returns the likelihood of X for bivariate Gaussian specified with mu and cov.

    X  ... vector to be evaluated -- np.array([[x_00, x_01], ..., [x_n0, x_n1]])
    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01],[cov_10, cov_11]]
    """

    dist = multivariate_normal(mu, cov)
    P = dist.pdf(X)
    return P


def classify_and_plot_data(X, r_mn, M):
    """ Performs soft or hard classification for EM and K-means algorithms respectively and plots the results

    :param X: data to be evaluated
    :param r_mn: classification matrix
    :param M: number of components
    :return: 
    """

    classif = np.argmax(r_mn, axis=0)

    for i in range(M):
        X_class = np.take(X, np.where(classif == i), axis=0)
        plot_values_and_mean(X_class)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Final GMM and respective means')
    plt.show()

def plot_values_and_mean(X):
    """ Plots the values and mean of X    
    :param X: data to be evaluated
    """
    plt.plot(X[0][:, 0], X[0][:, 1], 'o', alpha=0.1)
    plt.plot(np.mean(X[0][:, 0]), np.mean(X[0][:, 1]), 'x', color='black')
    plt.text(np.mean(X[0][:, 0]) + 10, np.mean(X[0][:, 1]) + 10, '$\mu$')


def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, title):
    """Show contour plot for bivariate Gaussian with given mu and cov in the range specified.

    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01], [cov_10, cov_11]]
    xmin, xmax, ymin, ymax ... range for plotting
    """

    npts = 500
    deltaX = (xmax - xmin) / npts
    deltaY = (ymax - ymin) / npts
    stdev = [0, 0]

    stdev[0] = np.sqrt(cov[0][0])
    stdev[1] = np.sqrt(cov[1][1])
    x = np.arange(xmin, xmax, deltaX)
    y = np.arange(ymin, ymax, deltaY)
    X, Y = np.meshgrid(x, y)

    Z = mlab.bivariate_normal(X, Y, stdev[0], stdev[1], mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+', alpha=0.5)  # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    # plt.show()