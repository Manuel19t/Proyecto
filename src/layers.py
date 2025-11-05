import numpy as np
from src.activations import ActivationFunction
from src.utils import initialize_weights

class Layer:
    
    def __init__(self, input_dim, output_dim, activation: str = "sigmoid"):
        self.W, self.b = initialize_weights(input_dim, output_dim, activation)
        self.activation = activation
        self.x = None
        self.z = None
        self.dW = np.zeros((output_dim, input_dim))
        self.db = np.zeros((1, output_dim))
    
    def forward(self, x):
        """Realiza la pasada hacia adelante de la capa."""
        self.x = x
        self.z = x @ self.W + self.b
        
        if self.activation == "sigmoid":
            return ActivationFunction.sigmoid(self.z)
        elif self.activation == "relu":
            return ActivationFunction.relu(self.z)
        elif self.activation == "softmax":
            return ActivationFunction.softmax(self.z)
        elif self.activation == "tanh":
            return ActivationFunction.tanh(self.z)
        else:
            return self.z  # Sin activación
    
    def backward(self, dA):
        if self.activation == "sigmoid":
            dZ = dA * ActivationFunction.sigmoid_grad(self.z)
        elif self.activation == "relu":
            dZ = dA * ActivationFunction.relu_grad(self.z)
        elif self.activation == "tanh":
            dZ = dA * ActivationFunction.tanh_grad(self.z)
        else:
            dZ = dA  # Sin activación
        m = self.x.shape[0]
        self.dW = (self.x.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        return dZ @ self.W.T
    
    def params(self):
        return {'W': self.W, 'b': self.b}
    
    def grads(self):
        return {'dW': self.dW, 'db': self.db}