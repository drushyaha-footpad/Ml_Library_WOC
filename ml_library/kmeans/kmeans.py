import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        """
        Compute k-means clustering.
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # 1. Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # 2. Assign samples to nearest centroid
            labels = self._assign_clusters(X)

            # 3. Update centroids
            new_centroids = self._update_centroids(X, labels)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        labels = []
        for x in X:
            distances = [np.linalg.norm(x - c) for c in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def _update_centroids(self, X, labels):
        centroids = []
        for i in range(self.n_clusters):
            # Points assigned to cluster i
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids.append(np.mean(cluster_points, axis=0))
            else:
                # Handle empty cluster: re-initialize randomly or keep same? 
                # Keeping same is simplest for now, or pick random point
                centroids.append(self.centroids[i])
        return np.array(centroids)

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        """
        X = np.array(X)
        return self._assign_clusters(X)
