"""
Demonstration script for different beta schedulers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from utils.schedulers import get_beta_scheduler, get_available_schedulers


def demo_schedulers():
    """
    Demonstrate different beta schedulers.
    """
    epochs = 50
    epoch_range = range(epochs)
    
    # Define scheduler configurations
    scheduler_configs = {
        'linear': {'max_beta': 1.0, 'warmup_epochs': 15},
        'cosine': {'max_beta': 1.0, 'warmup_epochs': 15},
        'exponential': {'max_beta': 1.0, 'warmup_epochs': 15, 'decay_rate': 0.1}
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each scheduler
    for i, (scheduler_type, params) in enumerate(scheduler_configs.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Create scheduler
            scheduler = get_beta_scheduler(scheduler_type, **params)
            
            # Generate beta values
            beta_values = [scheduler(epoch) for epoch in epoch_range]
            
            # Plot
            ax.plot(epoch_range, beta_values, 'b-', linewidth=2, marker='o', markersize=3)
            ax.set_title(f'{scheduler_type.title()} Beta Schedule')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Beta Value')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            # Add vertical line at warmup_epochs
            ax.axvline(x=params['warmup_epochs'], color='r', linestyle='--', alpha=0.7, label='Warmup End')
            ax.legend()
    
    # Compare all schedulers in the last subplot
    ax = axes[3]
    colors = ['blue', 'red', 'green']
    for i, (scheduler_type, params) in enumerate(scheduler_configs.items()):
        scheduler = get_beta_scheduler(scheduler_type, **params)
        beta_values = [scheduler(epoch) for epoch in epoch_range]
        ax.plot(epoch_range, beta_values, color=colors[i], linewidth=2, 
                label=f'{scheduler_type.title()}', marker='o', markersize=2)
    
    ax.set_title('All Beta Schedules Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Beta Value')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('beta_schedulers_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical comparison
    print("\nBeta Values Comparison (First 20 epochs):")
    print("Epoch", end="")
    for scheduler_type in scheduler_configs.keys():
        print(f"\t{scheduler_type.title()}", end="")
    print()
    
    for epoch in range(min(20, epochs)):
        print(f"{epoch:2d}", end="")
        for scheduler_type, params in scheduler_configs.items():
            scheduler = get_beta_scheduler(scheduler_type, **params)
            beta = scheduler(epoch)
            print(f"\t{beta:.3f}", end="")
        print()


def demo_custom_scheduler():
    """
    Demonstrate custom scheduler parameters.
    """
    epochs = 30
    epoch_range = range(epochs)
    
    # Different configurations for the same scheduler type
    configs = [
        {'max_beta': 1.0, 'warmup_epochs': 10},
        {'max_beta': 0.5, 'warmup_epochs': 10},
        {'max_beta': 1.0, 'warmup_epochs': 20},
        {'max_beta': 2.0, 'warmup_epochs': 15}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, config in enumerate(configs):
        ax = axes[i]
        scheduler = get_beta_scheduler('linear', **config)
        beta_values = [scheduler(epoch) for epoch in epoch_range]
        
        ax.plot(epoch_range, beta_values, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_title(f'Linear: max_beta={config["max_beta"]}, warmup={config["warmup_epochs"]}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Beta Value')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=config['warmup_epochs'], color='r', linestyle='--', alpha=0.7, label='Warmup End')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('custom_schedulers_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Available schedulers:", get_available_schedulers())
    print("\nDemonstrating different beta schedulers...")
    demo_schedulers()
    
    print("\nDemonstrating custom scheduler parameters...")
    demo_custom_scheduler()
    
    print("\nDemo completed! Check the generated PNG files for visualizations.") 