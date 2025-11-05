import numpy as np
from src.layers import Layer

class NeuralNetwork:
    
    def __init__(self, layers_dim, activation="sigmoid"):
        
        if isinstance(activation, str):
            activation = [activation] * (len(layers_dim) - 1)

        self.layers = []
        for i in range(len(layers_dim) - 1):
            self.layers.append(Layer(layers_dim[i], layers_dim[i + 1], activation=activation[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
        return loss_grad
    
    def params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.params()
            params.update({f'layer_{i+1}_{key}': value for key, value in layer_params.items()})
        return params
    
    def zeros_grads(self):
        for layer in self.layers:
            layer.dW.fill(0)
            layer.db.fill(0)