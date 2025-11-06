import numpy as np

class MSE:
    @staticmethod
    def forward(y_true, y_pred):
        diff = y_pred - y_true
        return np.mean(np.sum(diff * diff, axis=1) / 2.0)

    @staticmethod
    def backward(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]

class CrossEntropy:
    @staticmethod
    def forward(y_true, y_pred, eps=1e-9):
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def backward(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]

def get_loss(name):
    name = name.lower()
    if name in ("mse", "mean_squared_error"):
        return MSE
    if name in ("ce", "crossentropy", "cross-entropy", "cross_entropy"):
        return CrossEntropy
    raise ValueError(f"Unknown loss: {name}")
