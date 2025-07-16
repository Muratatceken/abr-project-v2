"""
Early stopping utilities for training.
"""


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs (patience).
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            min_delta (float): Minimum change in monitored quantity to qualify as improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_epoch = 0
        
    def step(self, val_loss: float, epoch: int = None) -> bool:
        """
        Update early stopping state with current validation loss.
        
        Args:
            val_loss (float): Current validation loss
            epoch (int, optional): Current epoch number for tracking
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = val_loss
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
        else:
            # No improvement
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_epoch = 0
    
    def __repr__(self) -> str:
        return f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, best_loss={self.best_loss:.6f})"


class ReduceLROnPlateau:
    """
    Reduce learning rate when validation loss plateaus.
    
    Complements early stopping by reducing learning rate before stopping training.
    """
    
    def __init__(self, patience: int = 5, factor: float = 0.5, min_lr: float = 1e-6, verbose: bool = True):
        """
        Initialize learning rate reduction on plateau.
        
        Args:
            patience (int): Number of epochs to wait before reducing LR
            factor (float): Factor by which to reduce learning rate
            min_lr (float): Minimum learning rate
            verbose (bool): Whether to print LR reduction messages
        """
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = float('inf')
        self.lr_reduced = False
        
    def step(self, val_loss: float, optimizer, epoch: int = None) -> bool:
        """
        Update state and potentially reduce learning rate.
        
        Args:
            val_loss (float): Current validation loss
            optimizer: PyTorch optimizer to modify
            epoch (int, optional): Current epoch for logging
            
        Returns:
            bool: True if learning rate was reduced, False otherwise
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.lr_reduced = False
        else:
            self.counter += 1
            
        if self.counter >= self.patience and not self.lr_reduced:
            # Reduce learning rate
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
                if self.verbose:
                    epoch_str = f" at epoch {epoch}" if epoch is not None else ""
                    print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}{epoch_str}")
                    
                self.lr_reduced = True
                self.counter = 0  # Reset counter after reduction
                return True
                
        return False
    
    def reset(self):
        """Reset state."""
        self.counter = 0
        self.best_loss = float('inf')
        self.lr_reduced = False
    
    def __repr__(self) -> str:
        return f"ReduceLROnPlateau(patience={self.patience}, factor={self.factor}, min_lr={self.min_lr})" 