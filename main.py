import numpy as np
import matplotlib.pyplot as plt
import random
from GMM import GMM

# Generate cluster centers, spread, and # of points.
center_1, center_2 = [i for i in [[3, 3],[7, 7]]]
cov_matrix = [[0.6, 0], [0, 0.6]]
N = 1000

# Generate and plot datapoints.
x1 = np.random.multivariate_normal(center_1, cov_matrix, N)
x2 = np.random.multivariate_normal(center_2, cov_matrix, N)

X = np.vstack((x1, x2))


# plot unlabelled data
plt.plot(X[: ,0] ,X[: ,1] ,'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Unlabelled data')
plt.show()

# plot all labels
plt.plot(x1[:, 0], x1[:, 1], 'o')
plt.plot(np.mean(x1[: ,0]) ,np.mean(x1[: ,1]) ,'x', color='black')
plt.text(np.mean(x1[: ,0]) , np.mean(x1[: ,1]), '$\mu_{a}$')

plt.plot(x2[:, 0], x2[:, 1], 'o')
plt.plot(np.mean(x2[:, 0]), np.mean(x2[:, 1]), 'x', color='black')
plt.text(np.mean(x2[:, 0]), np.mean(x2[:, 1]), '$\mu_{e}$')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Labelled data and corresponding means')
plt.show()

M = 2
max_iter = 200
tol = 1e-3
diagonal = False

gmm = GMM(X, M)

# run the K-means algorithm firstly to initialize the means of the GMM algorithm
# mu_0 = random.sample(list(X), M)
# mu_0, D = k_means(X, M, mu_0=mu_0, max_iter=max_iter, tol=tol, interactive=False)

# 1.) EM algorithm for GMM:
# TODO
L = gmm.EM(max_iter=max_iter, tol = tol, interactive=False, diagonal=False)

plt.ioff()
plt.plot(L)
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('EM log-likelihood function')
plt.show()


# 2.) K-means algorithm:
# TODO
#mu_0 = random.sample(list(X),M)
#D = gmm.k_means(max_iter=max_iter, tol=1e-2, interactive=False)

#plt.ioff()
#plt.plot(D)
#plt.xlabel('Iteration')
#plt.ylabel('Value')
#plt.title('Cumulative distance')
#plt.show()

# 3.) Sampling from GMM
# TODO
Y = gmm.sample(N=N)

for i in range(M):
    plt.plot(Y[i][:, 0], Y[i][:, 1], 'o')
    plt.plot(np.mean(Y[i][:, 0]), np.mean(Y[i][:, 1]), 'x', color='black')
    plt.text(np.mean(Y[i][:, 0]) + 10, np.mean(Y[i][:, 1]) + 10, '$\mu_{a}$')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Sampled data and corresponding means')
plt.show()