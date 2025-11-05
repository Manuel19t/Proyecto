import numpy as np


class ActivationFunction:
    
    @staticmethod
    def sigmoid(x):
        """Calcula la función de activación sigmoid."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_grad(x):
        """Calcula la derivada de la función de activación sigmoid."""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        """Calcula la función de activación ReLU."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x):
        """Calcula la derivada de la función de activación ReLU."""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        """Calcula la función de activación softmax."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)    
    
    @staticmethod
    def tanh(x):
        """Calcula la función de activación tanh."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_grad(x):
        """Calcula la derivada de la función de activación tanh."""
        return 1 - np.tanh(x) ** 2