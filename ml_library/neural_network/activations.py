import numpy as np

class Activation:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, d_out):
        pass

class ReLU(Activation):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self.input > 0)

class Sigmoid(Activation):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, d_out):
        return d_out * self.out * (1 - self.out)

class Tanh(Activation):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, d_out):
        return d_out * (1 - self.out ** 2)

class Softmax(Activation):
    def __init__(self):
        self.out = None

    def forward(self, x):
        # Stable softmax
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, d_out):
        # Softmax backward is often combined with CrossEntropy for stability,
        # but pure local Jacobian is complex for batched input output.
        # Here we assume it is passed dZ directly or implement simplified version
        # Usually we implement d_softmax separately or assume it's handled in loss.
        # For now return d_out assuming it's dL/dZ (combined) if used with CE
        # If used standalone (not common), we need full Jacobian.
        # Let's support the passed-through gradient assuming standard composition
        # OR implement pass-through for now as placeholder for CE integration.
        return d_out 
