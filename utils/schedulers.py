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


class CyclicalBetaScheduler:
    """
    Cyclical beta scheduler for periodic KL annealing.
    
    Implements cyclical annealing with configurable wave shape (cosine or triangle)
    between min_beta and max_beta over cycle_length epochs. This allows for
    periodic exploration of different KL regularization strengths.
    """
    
    def __init__(self, cycle_length: int = 20, min_beta: float = 0.0, max_beta: float = 1.0, 
                 wave_type: str = 'cosine'):
        """
        Initialize the cyclical beta scheduler.
        
        Args:
            cycle_length (int): Number of epochs per complete cycle
            min_beta (float): Minimum beta value in the cycle
            max_beta (float): Maximum beta value in the cycle
            wave_type (str): Type of wave ('cosine' or 'triangle')
        """
        self.cycle_length = cycle_length
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.wave_type = wave_type.lower()
        
        if self.wave_type not in ['cosine', 'triangle']:
            raise ValueError(f"Unsupported wave_type: {wave_type}. Use 'cosine' or 'triangle'")
    
    def __call__(self, epoch: int) -> float:
        """
        Get the beta value for the current epoch using cyclical schedule.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            
        Returns:
            float: Beta value for the current epoch
        """
        import math
        
        # Position within the current cycle [0, 1)
        cycle_position = (epoch % self.cycle_length) / self.cycle_length
        
        if self.wave_type == 'cosine':
            # Cosine wave: starts at max, goes to min, back to max
            beta_ratio = 0.5 * (1 + math.cos(2 * math.pi * cycle_position))
        elif self.wave_type == 'triangle':
            # Triangle wave: linear increase then decrease
            if cycle_position <= 0.5:
                # Ascending: min to max
                beta_ratio = 2 * cycle_position
            else:
                # Descending: max to min
                beta_ratio = 2 * (1 - cycle_position)
        
        # Scale to [min_beta, max_beta]
        return self.min_beta + beta_ratio * (self.max_beta - self.min_beta)
    
    def __repr__(self) -> str:
        return (f"CyclicalBetaScheduler(cycle_length={self.cycle_length}, "
                f"min_beta={self.min_beta}, max_beta={self.max_beta}, "
                f"wave_type='{self.wave_type}')")


class WarmRestartBetaScheduler:
    """
    Beta scheduler with warm restarts for periodic regularization reset.
    
    Combines any base scheduler with periodic restarts, resetting beta to 0
    every restart_interval epochs. This allows the model to periodically
    focus on reconstruction before re-introducing KL regularization.
    """
    
    def __init__(self, base_scheduler, restart_interval: int = 50, 
                 restart_multiplier: float = 1.0):
        """
        Initialize the warm restart beta scheduler.
        
        Args:
            base_scheduler: Base scheduler to apply within each restart cycle
            restart_interval (int): Number of epochs between restarts
            restart_multiplier (float): Multiplier for restart interval after each restart
        """
        self.base_scheduler = base_scheduler
        self.restart_interval = restart_interval
        self.restart_multiplier = restart_multiplier
        self.restart_epochs = [0]  # Track restart epoch numbers
        
    def __call__(self, epoch: int) -> float:
        """
        Get the beta value for the current epoch with warm restarts.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            
        Returns:
            float: Beta value for the current epoch
        """
        # Update restart schedule if needed
        while epoch >= self.restart_epochs[-1] + int(self.restart_interval * (self.restart_multiplier ** (len(self.restart_epochs) - 1))):
            next_restart = self.restart_epochs[-1] + int(self.restart_interval * (self.restart_multiplier ** (len(self.restart_epochs) - 1)))
            self.restart_epochs.append(next_restart)
        
        # Find the most recent restart
        current_restart_epoch = max([r for r in self.restart_epochs if r <= epoch])
        
        # Calculate epochs since last restart
        epochs_since_restart = epoch - current_restart_epoch
        
        # Apply base scheduler from the restart point
        return self.base_scheduler(epochs_since_restart)
    
    def get_restart_info(self, epoch: int):
        """Get information about restart status."""
        current_restart_epoch = max([r for r in self.restart_epochs if r <= epoch])
        epochs_since_restart = epoch - current_restart_epoch
        next_restart = min([r for r in self.restart_epochs if r > epoch] + [float('inf')])
        
        return {
            'last_restart_epoch': current_restart_epoch,
            'epochs_since_restart': epochs_since_restart,
            'next_restart_epoch': next_restart if next_restart != float('inf') else None,
            'is_restart_epoch': epoch in self.restart_epochs
        }
    
    def __repr__(self) -> str:
        return (f"WarmRestartBetaScheduler(base_scheduler={self.base_scheduler}, "
                f"restart_interval={self.restart_interval}, "
                f"restart_multiplier={self.restart_multiplier})")


class HierarchicalBetaScheduler:
    """
    Beta scheduler specifically designed for hierarchical CVAE with separate
    global and local KL weights that can evolve differently over training.
    """
    
    def __init__(self, global_scheduler, local_scheduler):
        """
        Initialize hierarchical beta scheduler.
        
        Args:
            global_scheduler: Scheduler for global KL weight
            local_scheduler: Scheduler for local KL weight
        """
        self.global_scheduler = global_scheduler
        self.local_scheduler = local_scheduler
    
    def __call__(self, epoch: int):
        """
        Get both global and local beta values for the current epoch.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            
        Returns:
            dict: Dictionary with 'global_kl_weight' and 'local_kl_weight'
        """
        return {
            'global_kl_weight': self.global_scheduler(epoch),
            'local_kl_weight': self.local_scheduler(epoch)
        }
    
    def __repr__(self) -> str:
        return (f"HierarchicalBetaScheduler(global_scheduler={self.global_scheduler}, "
                f"local_scheduler={self.local_scheduler})")


def get_beta_scheduler(scheduler_type: str, **kwargs):
    """
    Factory function to create beta schedulers based on type.
    
    Args:
        scheduler_type (str): Type of scheduler 
            ('linear', 'cosine', 'exponential', 'cyclical', 'warm_restart', 'hierarchical')
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
    elif scheduler_type == 'cyclical':
        return CyclicalBetaScheduler(**kwargs)
    elif scheduler_type == 'warm_restart':
        # Extract base scheduler config
        base_type = kwargs.pop('base_scheduler_type', 'cosine')
        base_kwargs = kwargs.pop('base_scheduler_kwargs', {})
        restart_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['restart_interval', 'restart_multiplier']}
        
        base_scheduler = get_beta_scheduler(base_type, **base_kwargs)
        return WarmRestartBetaScheduler(base_scheduler, **restart_kwargs)
    elif scheduler_type == 'hierarchical':
        # Extract global and local scheduler configs
        global_config = kwargs.get('global_scheduler', {'type': 'cosine', 'max_beta': 0.01})
        local_config = kwargs.get('local_scheduler', {'type': 'cosine', 'max_beta': 0.01})
        
        global_type = global_config.pop('type', 'cosine')
        local_type = local_config.pop('type', 'cosine')
        
        global_scheduler = get_beta_scheduler(global_type, **global_config)
        local_scheduler = get_beta_scheduler(local_type, **local_config)
        
        return HierarchicalBetaScheduler(global_scheduler, local_scheduler)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                        f"Supported types: {get_available_schedulers()}")


def get_available_schedulers():
    """
    Get list of available beta scheduler types.
    
    Returns:
        List of available scheduler type names
    """
    return ['linear', 'cosine', 'exponential', 'cyclical', 'warm_restart', 'hierarchical'] 