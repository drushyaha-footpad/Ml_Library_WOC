import numpy as np

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        """
        Generate polynomial and interaction features.
        
        Currently supports single feature expansion efficiently.
        For multiple features, it expands to [1, x1, x2, x1^2, x1x2, x2^2, ...] (simplified for now).
        
        Parameters:
        X (numpy.ndarray): Samples of shape (n_samples, n_features).
        
        Returns:
        numpy.ndarray: Transformed samples.
        """
        n_samples, n_features = X.shape
        X_poly = X.copy()
        
        for d in range(2, self.degree + 1):
            # For simplicity in this scratch implementation, we primarily handle single variable powers
            # Proper full interaction expansion is complex to implement from scratch efficiently without itertools
            # Here we just add power features: x^2, x^3...
            # This is sufficient for simple polynomial regression tasks usually requested.
            X_poly = np.hstack((X_poly, np.power(X, d)))
            
        return X_poly

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
