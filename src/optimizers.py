import numpy as np

class Optimizer:
    def __init__(self, lr):
        self.lr = lr
        
    def update(self, params, grads):
        raise NotImplementedError("This method should be overridden by subclasses.")

class Adam(Optimizer):
    def __init__(self, lr, beta1, beta2, epsilon):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
    

class SGD(Optimizer):
    
    def __init__(self, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
        
    def update(self, params, grads):
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            grad = grads[key] + self.weight_decay * params[key]
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
            
            params[key] += self.velocity[key]