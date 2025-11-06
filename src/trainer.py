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

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64, classification=True, verbose=True):
        history = {"train_loss": [], "val_loss": []}
        if classification:
            history["train_acc"] = []
            history["val_acc"] = []

        for epoch in range(1, epochs + 1):
            train_losses = []
            train_accs = []
            for Xb, yb in self._iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                out = self.network.forward(Xb)
                loss = self.loss_fn.forward(yb, out)
                train_losses.append(loss)
                grad_out = self.loss_fn.backward(yb, out)
                self.network.zero_grad()
                self.network.backward(grad_out)
                self.optimizer.update(self.network.params(), self.network.grads())
                if classification:
                    train_accs.append(accuracy(yb, out))

            history["train_loss"].append(float(np.mean(train_losses)))
            if classification:
                history["train_acc"].append(float(np.mean(train_accs)) if train_accs else 0.0)

            if X_val is not None and y_val is not None:
                val_out = self.network.forward(X_val)
                vloss = self.loss_fn.forward(y_val, val_out)
                history["val_loss"].append(float(vloss))
                if classification:
                    history["val_acc"].append(float(accuracy(y_val, val_out)))
            else:
                history["val_loss"].append(None)
                if classification:
                    history["val_acc"].append(None)

            if verbose:
                if classification:
                    print(f"Epoch {epoch:03d} | loss {history['train_loss'][-1]:.4f} | acc {history['train_acc'][-1]:.4f} | val_loss {history['val_loss'][-1]} | val_acc {history['val_acc'][-1]}")
                else:
                    print(f"Epoch {epoch:03d} | loss {history['train_loss'][-1]:.4f} | val_loss {history['val_loss'][-1]}")

        return history
