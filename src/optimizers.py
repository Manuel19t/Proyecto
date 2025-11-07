import numpy as np

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads):
        self.t += 1
        for p, g in zip(params, grads):
            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p)
                self.v[pid] = np.zeros_like(p)
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p
            m = self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * g
            v = self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SGD:
    def __init__(self, lr=1e-2, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}

    def update(self, params, grads):
        for p, g in zip(params, grads):
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p
            pid = id(p)
            v_prev = self.v.get(pid, np.zeros_like(p))
            v = self.momentum * v_prev - self.lr * g
            self.v[pid] = v
            p += v
