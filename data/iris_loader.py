import numpy as np
from pathlib import Path
from src.utils import split_data
from src.utils import preprocess_data
from src.utils import one_hot_encode

SCRIPT_DIR = Path(__file__).parent
DEFAULT_IRIS_PATH = SCRIPT_DIR / 'IRIS.csv'

def load_iris_data(path=DEFAULT_IRIS_PATH, 
                   train_rat=0.8, val_rat=0.1, test_rat=0.1, 
                   random_seed=None, shuffle=True):
    # Load the dataset from a CSV file
    data = np.loadtxt(path, delimiter=',', dtype=str, skiprows=1)
    X = data[:, :-1].astype(float)  # Features
    y = data[:, -1]                 # Labels
    
    class_labels = np.unique(y)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    y = np.array([label_to_index[label] for label in y])
    
    X = preprocess_data(X)
    y = one_hot_encode(y, len(class_labels))
    
    return split_data(X, y, train_rat, val_rat, test_rat, random_seed, shuffle)
    