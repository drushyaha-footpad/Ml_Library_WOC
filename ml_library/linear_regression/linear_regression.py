import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        """
        Linear Regression Model.
        
        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations for gradient descent.
        method (str): 'gradient_descent' or 'closed_form' (Normal Equation).
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
        X (numpy.ndarray): Training features of shape (n_samples, n_features).
        y (numpy.ndarray): Training labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if self.method == 'closed_form':
            self._fit_closed_form(X, y)
        else:
            self._fit_gradient_descent(X, y, n_samples)

    def _fit_gradient_descent(self, X, y, n_samples):
        for _ in range(self.n_iterations):
            # Forward pass (Prediction)
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and record loss (MSE)
            loss = np.mean((y_predicted - y) ** 2)
            self.losses.append(loss)

    def _fit_closed_form(self, X, y):
        # Add bias term (column of ones) to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal Equation: theta = (X.T * X)^-1 * X.T * y
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.bias = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X):
        """
        Predict target values for samples in X.
        
        Parameters:
        X (numpy.ndarray): Samples of shape (n_samples, n_features).
        
        Returns:
        numpy.ndarray: Predicted values.
        """
        return np.dot(X, self.weights) + self.bias
