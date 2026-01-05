import numpy as np
import sys
import os

# Add parent directory to path to allow importing custom_ml_library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_library.linear_model.linear_regression import LinearRegression
from ml_library.linear_model.logistic_regression import LogisticRegression
from ml_library.neighbors.knn import KNeighborsClassifier
from ml_library.cluster.kmeans import KMeans
from ml_library.tree.decision_tree import DecisionTreeClassifier

def test_linear_regression():
    print("Testing Linear Regression...")
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10]) # y = 2x
    
    lr = LinearRegression(n_iterations=2000, learning_rate=0.01)
    lr.fit(X, y)
    preds = lr.predict(np.array([[6]]))
    print(f"Prediction for 6 (expected ~12): {preds[0]:.2f}")
    assert abs(preds[0] - 12) < 0.1
    print("Linear Regression Passed!")

def test_logistic_regression():
    print("\nTesting Logistic Regression...")
    X = np.array([[1], [2], [3], [8], [9], [10]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    lr = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    lr.fit(X, y)
    preds = lr.predict(np.array([[2], [9]]))
    print(f"Predictions for 2, 9 (expected [0, 1]): {preds}")
    assert preds == [0, 1]
    print("Logistic Regression Passed!")

def test_knn():
    print("\nTesting KNN...")
    X = np.array([[1, 1], [1, 2], [5, 5], [5, 6]])
    y = np.array([0, 0, 1, 1])
    
    knn = KNeighborsClassifier(k=3)
    knn.fit(X, y)
    pred = knn.predict([[2, 2]])[0]
    print(f"Prediction for [2, 2] (expected 0): {pred}")
    assert pred == 0
    print("KNN Passed!")

def test_kmeans():
    print("\nTesting KMeans...")
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, max_iter=100)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    print(f"Labels: {labels}")
    # Should cluster first two together and last two together
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    print("KMeans Passed!")

def test_decision_tree():
    print("\nTesting Decision Tree...")
    X = np.array([[1, 2], [2, 3], [5, 5], [6, 6]])
    y = np.array([0, 0, 1, 1])
    
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X, y)
    preds = dt.predict(np.array([[1, 1], [7, 7]]))
    print(f"Predictions for [1,1], [7,7] (expected [0, 1]): {preds}")
    assert preds[0] == 0 and preds[1] == 1
    print("Decision Tree Passed!")

if __name__ == "__main__":
    test_linear_regression()
    test_logistic_regression()
    test_knn()
    test_kmeans()
    test_decision_tree()
