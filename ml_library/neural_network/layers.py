import numpy as np

class Layer:
    def __init__(self):
        self.params = []
        self.grads = []

    def forward(self, input):
        pass
    
    def backward(self, grad_output):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size, initialization='he'):
        super().__init__()
        self.input = None
        
        # Weight Initialization
        if initialization == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        elif initialization == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        else: # Random
            self.weights = np.random.randn(input_size, output_size) * 0.01
            
        self.bias = np.zeros((1, output_size))
        
        self.params = [self.weights, self.bias]
        self.grads = [np.zeros_like(self.weights), np.zeros_like(self.bias)]

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, grad_output):
        # Gradients wrt parameters
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        self.grads = [grad_weights, grad_bias]
        
        # Gradient wrt input
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

class BatchNorm1d(Layer):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Trainable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        
        # Running stats
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        self.training = True
        self.cache = None

    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False

    def forward(self, input):
        if self.training:
            batch_mean = np.mean(input, axis=0, keepdims=True)
            batch_var = np.var(input, axis=0, keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            normalized = (input - batch_mean) / np.sqrt(batch_var + self.epsilon)
            self.cache = (input, normalized, batch_mean, batch_var)
        else:
            normalized = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            
        return self.gamma * normalized + self.beta

    def backward(self, grad_output):
        if not self.training:
            # Simplified backward for eval (not typical but for completeness)
            return grad_output
            
        input, normalized, mean, var = self.cache
        N, D = grad_output.shape
        
        # Gradients wrt params
        grad_gamma = np.sum(grad_output * normalized, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        self.grads = [grad_gamma, grad_beta]
        
        # Gradient wrt input
        dnormalized = grad_output * self.gamma
        ivar = 1 / np.sqrt(var + self.epsilon)
        
        dvar = np.sum(dnormalized * (input - mean) * -0.5 * (var + self.epsilon)**(-1.5), axis=0, keepdims=True)
        dmean = np.sum(dnormalized * -ivar, axis=0, keepdims=True) + dvar * np.mean(-2 * (input - mean), axis=0, keepdims=True)
        
        grad_input = (dnormalized * ivar) + (dvar * 2 * (input - mean) / N) + (dmean / N)
        return grad_input
