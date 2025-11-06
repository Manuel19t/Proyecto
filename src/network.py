import numpy as np
from .layers import Dense

class NeuralNetwork:
    def __init__(self, architecture, activations):
        assert len(activations) == len(architecture) - 1
        self.layers = []
        for i in range(len(architecture) - 1):
            self.layers.append(Dense(architecture[i], architecture[i + 1], activation=activations[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_loss_out):
        grad = grad_loss_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def grads(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.grads())
        return grads

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
