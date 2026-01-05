import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeBase:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
             # No split found likely due to identical features with different labels or numerical stability
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # Simple optimization: Randomly sample thresholds if too many
            if len(thresholds) > 100:
                thresholds = np.random.choice(thresholds, 100, replace=False)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Parent loss
        parent_loss = self._calculate_loss(y)

        # Generate split
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Weighted child loss
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._calculate_loss(y[left_idxs]), self._calculate_loss(y[right_idxs])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # Information Gain is difference in loss
        ig = parent_loss - child_loss
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_loss(self, y):
        # Gini for Classification, MSE/Variance for Regression
        raise NotImplementedError

    def _calculate_leaf_value(self, y):
        raise NotImplementedError

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class DecisionTreeClassifier(DecisionTreeBase):
    def _calculate_loss(self, y):
        # Gini Impurity
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = len(y[y == cls]) / len(y)
            gini -= p ** 2
        return gini

    def _calculate_leaf_value(self, y):
        # Most common class label
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 0: return 0 # Handle empty case safety
        return unique[np.argmax(counts)]

class DecisionTreeRegressor(DecisionTreeBase):
    def _calculate_loss(self, y):
        # Mean Squared Error or Variance
        if len(y) == 0: return 0
        return np.var(y)

    def _calculate_leaf_value(self, y):
        # Mean value
        if len(y) == 0: return 0
        return np.mean(y)
