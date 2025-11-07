import numpy as np
import matplotlib.pyplot as plt


def shuffle_data(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def split_data(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=None, shuffle=True):
    if random_seed is not None:
        np.random.seed(random_seed)
    if shuffle:
        X, y = shuffle_data(X, y)
    N = X.shape[0]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
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
    return X_min, X_max, denom


def transform_minmax(X, X_min, denom):
    return (X - X_min) / denom


def accuracy(y_true_onehot, y_pred_probs):
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return np.mean(y_true == y_pred)


def plot_curves(history, title_prefix=""):
    epochs = len(history["train_loss"])
    x = np.arange(epochs)
    
    if "train_acc" in history:
        fig, axs = plt.subplots(2, 1, figsize=(7, 7))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(7, 5))
        axs = [axs]  # para tratar axs[0] igual

    axs[0].plot(x, history["train_loss"], label="train_loss")
    if history["val_loss"] is not None and \
       all(v is not None for v in history["val_loss"]):
        axs[0].plot(x, history["val_loss"], label="val_loss")
    axs[0].set_title(f"{title_prefix} Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    if "train_acc" in history:
        axs[1].plot(x, history["train_acc"], label="train_acc")
        if "val_acc" in history:
            axs[1].plot(x, history["val_acc"], label="val_acc")
        axs[1].set_title(f"{title_prefix} Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()

    plt.tight_layout()
    plt.show()

def confusion_matrix(y_true_onehot, y_pred_probs, num_classes):
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.show()


def show_predictions(images, y_true, y_pred, labels, rows=3, cols=3):
    fig, axs = plt.subplots(rows, cols, figsize=(7, 7))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            axs[r, c].imshow(images[idx].reshape(28, 28), cmap="gray")
            axs[r, c].set_title(f"T:{labels[y_true[idx]]} P:{labels[y_pred[idx]]}")
            axs[r, c].axis("off")
            idx += 1
    plt.show()

def plot_regression_predictions(y_true, y_pred, title="Regression Predictions"):

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)
    
    # Línea de referencia perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.tight_layout()
    plt.show()