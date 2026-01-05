import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_library.neural_network import NeuralNetwork, Dense, ReLU, Sigmoid, Tanh, MSE, SGD, Adam

def test_xor():
    print("Testing Neural Network on XOR...")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    model = NeuralNetwork()
    model.add(Dense(2, 4))
    model.add(Tanh())
    model.add(Dense(4, 1))
    model.add(Sigmoid())
    
    optimizer = Adam(learning_rate=0.05)
    loss = MSE()
    
    model.compile(loss, optimizer)
    model.fit(X, y, epochs=3000, verbose=True)
    
    preds = model.predict(X)
    print(f"Predictions:\n{preds}")
    
    # Check simple correctness
    preds_binary = (preds > 0.5).astype(int)
    assert np.all(preds_binary == y)
    print("XOR Test Passed!")

if __name__ == "__main__":
    test_xor()
