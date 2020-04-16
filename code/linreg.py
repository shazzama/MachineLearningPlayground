import numpy as np
from scipy import stats
import utils


class LinearRegression:
    def __init__(self, p):
        self.p = p

    def fit(self, X, y):
        Z = self.__newBasis(X)
        self.w = np.linalg.lstsq(Z.T@Z, Z.T@y)

    def predict(self, Xtest):
        Z = self.__newBasis(Xtest)
        return Z@self.w[0]

    def __newBasis(self, X):
        n, d = X.shape
        Z = np.ones((n, d * self.p))
        for i in range(2, self.p +1):
            new_col = np.power(X, i)
            Z[:,d*(i-1):d*i] = new_col
        return Z
