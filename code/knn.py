import numpy as np
from scipy import stats
import utils


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, Xtest):
        T, D = Xtest.shape
        y_pred = np.zeros(T)
        print("DIST")
        dists = utils.euclidean_dist_squared(self.X, Xtest)
        print("FOR")
        for t in range(T):
            indices = np.argsort(dists[:, t])
            closest_k_indices = indices[:self.k + 1]

            labels = self.y[closest_k_indices]
            y_pred[t] = stats.mode(labels, axis=0)[0]
        return y_pred
