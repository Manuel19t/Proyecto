import numpy as np
from .losses import get_loss
from .utils import accuracy

class Trainer:
    def __init__(self, network, optimizer, loss="cross_entropy"):
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = get_loss(loss)

    def _iterate_minibatches(self, X, y, batch_size, shuffle=True):
        N = X.shape[0]
        indices = np.arange(N)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, 
              batch_size=64, classification=True, verbose=True, early_stopping=False, 
              patience=5, min_delta=0.0):
        history = {"train_loss": [], "val_loss": []}
        if classification:
            history["train_acc"] = []
            history["val_acc"] = []
            
        best_val_loss = float('inf')
        no_improve_epochs = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            train_losses = []
            train_accs = []
            for Xb, yb in self._iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                out = self.network.forward(Xb, training=True)
                loss = self.loss_fn.forward(yb, out)
                train_losses.append(loss)
                grad_out = self.loss_fn.backward(yb, out)
                self.network.zero_grad()
                self.network.backward(grad_out)
                self.optimizer.update(self.network.params(), self.network.grads())
                if classification:
                    train_accs.append(accuracy(yb, out))
                                                
            train_loss = float(np.mean(train_losses))
            history["train_loss"].append(train_loss)
            if classification:
                history["train_acc"].append(float(np.mean(train_accs)) if train_accs else 0.0)

            val_loss = None
            val_acc = None
            if X_val is not None and y_val is not None:
                val_out = self.network.forward(X_val, training=False)
                val_loss = float(self.loss_fn.forward(y_val, val_out))
                history["val_loss"].append(val_loss)
                if classification:
                    val_acc = float(accuracy(y_val, val_out))
                    history["val_acc"].append(val_acc)
            else:
                history["val_loss"].append(None)
                if classification:
                    history["val_acc"].append(None)

            if verbose:
                if classification:
                    print(f"Epoch {epoch:03d} | " +
                          f"loss {history['train_loss'][-1]:.4f} | acc {history['train_acc'][-1]:.4f} | " +
                          f"val_loss {history['val_loss'][-1]} | val_acc {history['val_acc'][-1]}")
                else:
                    print(f"Epoch {epoch:03d} | loss {history['train_loss'][-1]:.4f} | val_loss {history['val_loss'][-1]}")
            
            if early_stopping and X_val is not None and y_val is not None:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                    best_weights = [param.copy() for param in self.network.params()]
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        for param, best_param in zip(self.network.params(), best_weights):
                            np.copyto(param, best_param)
                        break
        return history
