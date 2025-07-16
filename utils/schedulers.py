"""
Schedulers for training hyperparameters.
"""


class BetaScheduler:
    """
    Beta scheduler for KL annealing in VAE training.
    
    Implements linear annealing from 0 to max_beta over warmup_epochs,
    then maintains max_beta for the remainder of training.
    """
    
    def __init__(self, max_beta: float = 1.0, warmup_epochs: int = 10):
        """
        Initialize the beta scheduler.
        
        Args:
            max_beta (float): Maximum beta value to reach after warmup
            warmup_epochs (int): Number of epochs over which to linearly increase beta
        """
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        
    def __call__(self, epoch: int) -> float:
        """
        Get the beta value for the current epoch.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            
        Returns:
            float: Beta value for the current epoch
        """
        if epoch >= self.warmup_epochs:
            return self.max_beta
        return (epoch / self.warmup_epochs) * self.max_beta
    
    def __repr__(self) -> str:
        return f"BetaScheduler(max_beta={self.max_beta}, warmup_epochs={self.warmup_epochs})"


class CosineAnnealingBetaScheduler:
    """
    Cosine annealing beta scheduler for more gradual transitions.
    
    Implements cosine annealing from 0 to max_beta over warmup_epochs,
    providing smoother transitions compared to linear annealing.
    """
    
    def __init__(self, max_beta: float = 1.0, warmup_epochs: int = 10):
        """
        Initialize the cosine annealing beta scheduler.
        
        Args:
            max_beta (float): Maximum beta value to reach after warmup
            warmup_epochs (int): Number of epochs over which to apply cosine annealing
        """
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        
    def __call__(self, epoch: int) -> float:
        """
        Get the beta value for the current epoch using cosine annealing.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            
        Returns:
            float: Beta value for the current epoch
        """
        import math
        
        if epoch >= self.warmup_epochs:
            return self.max_beta
        
        # Cosine annealing: 0.5 * (1 - cos(pi * epoch / warmup_epochs))
        progress = epoch / self.warmup_epochs
        return self.max_beta * 0.5 * (1 - math.cos(math.pi * progress))
    
    def __repr__(self) -> str:
        return f"CosineAnnealingBetaScheduler(max_beta={self.max_beta}, warmup_epochs={self.warmup_epochs})"


class ExponentialBetaScheduler:
    """
    Exponential beta scheduler for rapid initial increase followed by gradual approach.
    
    Implements exponential approach to max_beta over warmup_epochs.
    """
    
    def __init__(self, max_beta: float = 1.0, warmup_epochs: int = 10, decay_rate: float = 0.1):
        """
        Initialize the exponential beta scheduler.
        
        Args:
            max_beta (float): Maximum beta value to approach
            warmup_epochs (int): Number of epochs over which to apply exponential schedule
            decay_rate (float): Controls the rate of exponential approach (smaller = faster)
        """
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        
    def __call__(self, epoch: int) -> float:
        """
        Get the beta value for the current epoch using exponential schedule.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            
        Returns:
            float: Beta value for the current epoch
        """
        import math
        
        if epoch >= self.warmup_epochs:
            return self.max_beta
        
        # Exponential approach: max_beta * (1 - exp(-decay_rate * epoch))
        return self.max_beta * (1 - math.exp(-self.decay_rate * epoch))
    
    def __repr__(self) -> str:
        return f"ExponentialBetaScheduler(max_beta={self.max_beta}, warmup_epochs={self.warmup_epochs}, decay_rate={self.decay_rate})"


def get_beta_scheduler(scheduler_type: str, **kwargs):
    """
    Factory function to create beta schedulers based on type.
    
    Args:
        scheduler_type (str): Type of scheduler ('linear', 'cosine', 'exponential')
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        Beta scheduler instance
        
    Raises:
        ValueError: If scheduler_type is not supported
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'linear':
        return BetaScheduler(**kwargs)
    elif scheduler_type == 'cosine':
        return CosineAnnealingBetaScheduler(**kwargs)
    elif scheduler_type == 'exponential':
        return ExponentialBetaScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                        f"Supported types: 'linear', 'cosine', 'exponential'")


def get_available_schedulers():
    """
    Get list of available beta scheduler types.
    
    Returns:
        List of available scheduler type names
    """
    return ['linear', 'cosine', 'exponential'] 