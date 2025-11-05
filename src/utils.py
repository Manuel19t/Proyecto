import numpy as np 
import matplotlib.pyplot as plt
from src.activations import ActivationFunction

def initialize_weights(input_dim, output_dim, activation: str = "sigmoid"):
    """Inicializa los pesos de una capa con distribución normal."""
    if activation in ["sigmoid", "tanh"]:
        limit = np.sqrt(6 / (input_dim + output_dim))
    elif activation == "relu":
        limit = np.sqrt(2 / input_dim)
    else:
        limit = 0.01  # Valor por defecto para otras activaciones
    weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
    biases = np.zeros((1, output_dim))
    return weights, biases

def shuffle_data(X, y):
    """Mezcla los datos de entrada y las etiquetas de manera sincronizada."""
    assert len(X) == len(y), "Las dimensiones de X e y deben coincidir."
    permutation = np.random.permutation(len(X))
    return X[permutation], y[permutation]

def split_data(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, ramdom_seed=None, shuffle=True):
    """Divide los datos en conjuntos de entrenamiento, validación y test."""
    assert train_ratio + val_ratio + test_ratio == 1.0, "La suma de los tamaños debe ser 1"
    
    if ramdom_seed is not None:
        np.random.seed(ramdom_seed)
    
    if shuffle:
        X, y = shuffle_data(X, y)
    
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_data(X):
    """Preprocesa los datos de entrada."""
    X = X.astype(np.float32)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    minus = X_max - X_min
    minus[minus == 0] = 1e-8  # Evitar división por cero
    return (X - X_min) / minus

def one_hot_encode(y, num_classes):
    """Convierte las etiquetas a codificación one-hot."""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def plot_loss_curve(history):
    """Grafica la curva de pérdida a lo largo de las épocas."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Pérdida de Entrenamiento')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Pérdida de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Curva de Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()