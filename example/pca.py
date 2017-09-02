import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../data/pcaData.txt')

# raw data plot
plt.figure('pca_raw_data')
plt.scatter(data[0, :], data[1, :])
#plt.show()


sigma = np.dot(data, data.T) / data.shape[1]
U, S, V = np.linalg.svd(sigma)
plt.figure('pca_raw_data_with_eigenvector')
plt.scatter(data[0, :], data[1, :])
plt.plot((0, U[0, 0]), (0, U[0, 1]))
plt.plot((0, U[1, 0]), (0, U[1, 1]))


data_rotate = np.dot(U.T, data)
plt.figure('pca_rotate_data')
plt.scatter(data_rotate[0, :], data_rotate[1, :])


data_with_one_dim = data_rotate[0, :]
plt.figure('pca_rotate_data')
plt.scatter(data_with_one_dim, np.zeros(data_with_one_dim.shape))

plt.show()