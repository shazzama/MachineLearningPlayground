import numpy as np
from scipy import stats
import utils


class LinearRegression:
    def __init__(self, p):
        self.p = p

    def fit(self, X, y):
        Z = self.__newBasis(X)
        self.w = np.linalg.lstsq(Z.T@Z, Z.T@y)
        print("SELF w")
        print(self.w[0].shape)

    def predict(self, Xtest):
        Z = self.__newBasis(Xtest)
        print("Ztest shape")
        print(Z.shape)
        return Z@self.w[0]

    def __newBasis(self, X):
        n, d = X.shape
        Z = np.ones((n, d+1))
        Z[:,1:] = X
        return Z
