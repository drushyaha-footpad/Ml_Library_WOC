import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Logistic Regression Classifier.
        
        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations for optimization.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
        X (numpy.ndarray): Training features of shape (n_samples, n_features).
        y (numpy.ndarray): Training labels (0 or 1) of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute loss (Binary Cross Entropy)
            # Clip predictions to avoid log(0) error
            y_pred_clipped = np.clip(y_predicted, 1e-15, 1 - 1e-15)
            loss = -(1 / n_samples) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.losses.append(loss)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X (numpy.ndarray): Samples of shape (n_samples, n_features).
        
        Returns:
        numpy.ndarray: Predicted class labels (0 or 1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

    def predict_proba(self, X):
        """
        Predict probability estimates.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
