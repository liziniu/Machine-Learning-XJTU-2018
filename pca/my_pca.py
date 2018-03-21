from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
images, targets = digits.data, digits.target
images_ = images.T


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.x = None
        self.x_mean = None
        self.x_reduce_mean = None
        self.eig_values = None
        self.eig_vectors = None
        self.res = None
        pass

    def fit(self, x):
        self.x = x
        self.x_reduce_mean = self._reduce_mean()
        self.eig_values, self.eig_vectors = self._get_eig()
        values_selected, vectors_selected = self.eig_vectors[:self.n_components], self.eig_vectors[:, :self.n_components]
        self.res = np.dot(vectors_selected.T, self.x_reduce_mean)
        return self.res

    def _reduce_mean(self):
        self.x_mean = np.mean(self.x, axis=1).reshape(-1, 1)
        return self.x - self.x_mean

    def _get_eig(self):
        self.cov = np.dot(self.x_reduce_mean, self.x_reduce_mean.T)
        eig_val, eig_vec = np.linalg.eig(self.cov)
        return eig_val, eig_vec

    def visualize(self, y=None):
        assert self.res.shape[0] == 2, "The reduced dimensionality must be 2!"
        plt.figure()
        plt.scatter(self.res[0, :], self.res[1, :], c=y)
        plt.show()

if __name__ == "__main__":
    pca = PCA(n_components=2)
    res = pca.fit(images_)
    pca.visualize(y=targets)
