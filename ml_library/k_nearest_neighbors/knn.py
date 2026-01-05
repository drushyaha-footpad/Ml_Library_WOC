import numpy as np

class KNeighborsBase:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_neighbors(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return k_nearest_labels

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        raise NotImplementedError("Subclasses must implement _predict_single")

class KNeighborsClassifier(KNeighborsBase):
    def _predict_single(self, x):
        k_nearest_labels = self._get_neighbors(x)
        # Return the most common class label
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]

class KNeighborsRegressor(KNeighborsBase):
    def _predict_single(self, x):
        k_nearest_labels = self._get_neighbors(x)
        # Return the mean of labels
        return np.mean(k_nearest_labels)
