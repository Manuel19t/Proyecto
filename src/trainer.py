import numpy as np
from src.utils import plot_loss_curve

class Trainer:
    
    def __init__(self, model, optimizer, loss_fn, loss_grad):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_grad = loss_grad
        
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=64):
        history = {"train_loss": [], "val_loss": []}
        num_samples = x_train.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            x_train = x_train[perm]
            y_train = y_train[perm]
            total_loss = 0
            batches = 0
            
            for i in range(0, num_samples, batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                y_pred = self.model.forward(x_batch)
                loss = self.loss_fn(y_batch, y_pred)
                total_loss += loss
                batches += 1

                loss_gradient = self.loss_grad(y_batch, y_pred)
                self.model.backward(loss_gradient)
                
                params, grads = {}, {}
                
                for j, layer in enumerate(self.model.layers):
                    params[f"W{j}"] = layer.W
                    params[f"b{j}"] = layer.b
                    grads[f"W{j}"] = layer.dW
                    grads[f"b{j}"] = layer.db
                
                self.optimizer.update(params, grads)
            
            avg_loss = total_loss / batches
            history["train_loss"].append(avg_loss)
            
            if x_val is not None and y_val is not None:
                y_val_pred = self.model.forward(x_val)
                val_loss = self.loss_fn(y_val, y_val_pred)
                history["val_loss"].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        plot_loss_curve(history)
        
        return history
    