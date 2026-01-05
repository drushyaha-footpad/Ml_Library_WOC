from .linear_regression import LinearRegression
from ..utils.preprocessing import PolynomialFeatures
import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000, method='closed_form'):
        """
        Polynomial Regression Model.
        
        Wraps PolynomialFeatures and LinearRegression.
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations, method=method)

    def fit(self, X, y):
        # transform X
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self.poly.fit_transform(X)
        return self.model.predict(X_poly)
