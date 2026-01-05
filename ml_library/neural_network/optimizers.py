import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def step(self, layers):
        for layer in layers:
            # Check if layer has params
            if not hasattr(layer, 'params'): continue
            
            for i in range(len(layer.params)):
                layer.params[i] -= self.learning_rate * layer.grads[i]

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocities = {} # Map layer_id -> list of velocities

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, 'params'): continue
            
            if id(layer) not in self.velocities:
                self.velocities[id(layer)] = [np.zeros_like(p) for p in layer.params]
                
            velocities = self.velocities[id(layer)]
            
            for i in range(len(layer.params)):
                velocities[i] = self.beta * velocities[i] + (1 - self.beta) * layer.grads[i]
                layer.params[i] -= self.learning_rate * velocities[i]

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.caches = {} 

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, 'params'): continue
            
            if id(layer) not in self.caches:
                self.caches[id(layer)] = [np.zeros_like(p) for p in layer.params]

            caches = self.caches[id(layer)]
            
            for i in range(len(layer.params)):
                 caches[i] = self.beta * caches[i] + (1 - self.beta) * (layer.grads[i] ** 2)
                 layer.params[i] -= self.learning_rate * layer.grads[i] / (np.sqrt(caches[i]) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, 'params'): continue
            
            if id(layer) not in self.m:
                self.m[id(layer)] = [np.zeros_like(p) for p in layer.params]
                self.v[id(layer)] = [np.zeros_like(p) for p in layer.params]

            ms = self.m[id(layer)]
            vs = self.v[id(layer)]
            
            for i in range(len(layer.params)):
                # Update biased first moment estimate
                ms[i] = self.beta1 * ms[i] + (1 - self.beta1) * layer.grads[i]
                # Update biased second raw moment estimate
                vs[i] = self.beta2 * vs[i] + (1 - self.beta2) * (layer.grads[i] ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = ms[i] / (1 - self.beta1 ** self.t)
                # Compute bias-corrected second raw moment estimate
                v_hat = vs[i] / (1 - self.beta2 ** self.t)
                
                layer.params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
