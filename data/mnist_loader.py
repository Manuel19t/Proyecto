import numpy as np
from pathlib import Path
from src.utils import split_data
from src.utils import preprocess_data
from src.utils import one_hot_encode

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MNIST_TEST_PATH = SCRIPT_DIR / 'fashion-mnist_test.csv'
DEFAULT_MNIST_TRAIN_PATH = SCRIPT_DIR / 'fashion-mnist_train.csv'

def load_mnist_data(train_path=DEFAULT_MNIST_TRAIN_PATH,
                    test_path=DEFAULT_MNIST_TEST_PATH, 
                    train_rat=0.8, val_rat=0.1, test_rat=0.1,
                    random_seed=None, shuffle=True):
    
    train_data = np.loadtxt(train_path, delimiter=',', dtype=int, skiprows=1)
    test_data = np.loadtxt(test_path, delimiter=',', dtype=int, skiprows=1)
    data = np.vstack((train_data, test_data)) # Combina los datos de entrenamiento y prueba
    
    X = data[:, 1:].astype(int)  # Features
    y = data[:, 0].astype(int) # Labels
    
    X = preprocess_data(X)
    y = one_hot_encode(y, 10)
    
    return split_data(X, y, train_rat, val_rat, test_rat, random_seed, shuffle)