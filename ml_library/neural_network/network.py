import numpy as np
try:
    from .layers import Layer
except ImportError:
    # Handle direct execution if needed
    pass

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss_func = loss
        self.optimizer = optimizer

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                y_pred = self.forward(X_batch)
                
                # Loss
                loss = self.loss_func.forward(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward
                grad = self.loss_func.backward(y_pred, y_batch)
                self.backward(grad)
                
                # Update
                self.optimizer.step(self.layers)
                
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (n_samples / batch_size):.4f}")

    def predict(self, X):
        return self.forward(X)
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'): layer.train()
            
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'): layer.eval()
