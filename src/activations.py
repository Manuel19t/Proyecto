import numpy as np

class Sigmoid:
    name = "sigmoid"
    @staticmethod
    def forward(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    @staticmethod
    def backward(a, grad_a):
        return grad_a * a * (1.0 - a)

class Tanh:
    name = "tanh"
    @staticmethod
    def forward(z):
        return np.tanh(z)
    @staticmethod
    def backward(a, grad_a):
        return grad_a * (1.0 - a**2)

class ReLU:
    name = "relu"
    @staticmethod
    def forward(z):
        return np.maximum(0.0, z)
    @staticmethod
    def backward(a, grad_a):
        return grad_a * (a > 0).astype(a.dtype)

class Softmax:
    name = "softmax"
    @staticmethod
    def forward(z):
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    @staticmethod
    def backward(a, grad_a):
        batch, n = a.shape
        grad_z = np.empty_like(a)
        for i in range(batch):
            s = a[i].reshape(-1, 1)
            J = np.diagflat(s) - s @ s.T
            grad_z[i] = J @ grad_a[i]
        return grad_z

def get_activation(name):
    name = name.lower()
    if name == "sigmoid":
        return Sigmoid
    if name == "tanh":
        return Tanh
    if name == "relu":
        return ReLU
    if name == "softmax":
        return Softmax
    raise ValueError(f"Unknown activation: {name}")
