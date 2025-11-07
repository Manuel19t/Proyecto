import numpy as np

class Linear:
    name = "linear"
    @staticmethod
    def forward(z):
        return z
    @staticmethod
    def backward(a, grad_a):
        return grad_a

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
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    @staticmethod
    def backward(a, grad_a):
        return grad_a

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
    if name == "linear":
        return Linear
    raise ValueError(f"Unknown activation: {name}")
