import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        pass
    
    def backward(self, y_pred, y_true):
        pass

class MSE(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        n_samples = y_pred.shape[0]
        return 2 * (y_pred - y_true) / n_samples

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # y_true should be one-hot encoded or indices? Assuming one-hot for now as standard
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

    def backward(self, y_pred, y_true):
        # Assuming y_pred came from Softmax
        # If it is separate:
        # dL/dy_pred = -y_true/y_pred
        # But usually we compute dL/dZ = y_pred - y_true (if Softmax included)
        # Here, strictly pure backward:
        n_samples = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) / n_samples
