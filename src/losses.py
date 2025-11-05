import numpy as np

class LossFunction:
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-9))
    
    @staticmethod
    def cross_entropy_grad(y_true, y_pred):
        return - (y_true / (y_pred + 1e-9)) / y_true.shape[0]

    @staticmethod
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_grad(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.shape[0]
