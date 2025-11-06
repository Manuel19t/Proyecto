import numpy as np

def shuffle_data(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def split_data(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=None, shuffle=True):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    if random_seed is not None:
        np.random.seed(random_seed)
    if shuffle:
        X, y = shuffle_data(X, y)
    N = X.shape[0]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def one_hot(y, num_classes):
    y = y.astype(int)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

def fit_minmax(X):
    X = X.astype(np.float32)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = np.where((X_max - X_min) == 0, 1e-8, (X_max - X_min))
    return X_min, denom

def transform_minmax(X, X_min, denom):
    return (X - X_min) / denom

def accuracy(y_true_onehot, y_pred_probs):
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return np.mean(y_true == y_pred)
