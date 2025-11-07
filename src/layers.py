import numpy as np
from .activations import get_activation

def he_init(input_dim, output_dim):
    std = np.sqrt(2.0 / input_dim)
    return np.random.randn(input_dim, output_dim) * std, np.zeros((1, output_dim))

def xavier_init(input_dim, output_dim):
    std = np.sqrt(1.0 / input_dim)
    return np.random.randn(input_dim, output_dim) * std, np.zeros((1, output_dim))

def initialize_weights(input_dim, output_dim, activation):
    if activation.lower() == "relu":
        return he_init(input_dim, output_dim)
    elif activation.lower() == "linear":
        return np.random.randn(input_dim, output_dim) * 0.01, np.zeros((1, output_dim))
    else:
        return xavier_init(input_dim, output_dim)

class Dense:
    def __init__(self, input_dim, output_dim, activation="sigmoid", dropout_rate=0.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.W, self.b = initialize_weights(input_dim, output_dim, activation)
        self.activation = get_activation(activation)
        self.dropout_rate = dropout_rate
        self.mask = None
        self.x = None
        self.z = None
        self.a = None
        self.dW = np.zeros((input_dim, output_dim))
        self.db = np.zeros((1, output_dim))

    def forward(self, x, training=True):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.activation.forward(self.z)
        
        if training and self.dropout_rate > 0.0:
            self.mask = (np.random.rand(*self.a.shape) > self.dropout_rate).astype(np.float32)
            self.a *= self.mask
            self.a /= (1.0 - self.dropout_rate)
        else:
            self.mask = None
        
        return self.a

    def backward(self, grad_a):
        if self.dropout_rate > 0.0 and self.mask is not None:
            grad_a = grad_a * self.mask
            grad_a /= (1.0 - self.dropout_rate)
        grad_z = self.activation.backward(self.a, grad_a)
        self.dW = self.x.T @ grad_z
        self.db = np.sum(grad_z, axis=0, keepdims=True)
        grad_x = grad_z @ self.W.T
        return grad_x

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)
