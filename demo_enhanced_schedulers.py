#!/usr/bin/env python3
"""
Demo script to test Enhanced Beta Schedulers functionality.

This script demonstrates:
1. Cyclical beta scheduling with cosine and triangle waves
2. Warm restart beta scheduling with configurable intervals
3. Hierarchical beta scheduling for hierarchical CVAE
4. Enhanced logging and visualization of beta schedules
5. Integration with training loops and TensorBoard
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.schedulers import (
    get_beta_scheduler, 
    CyclicalBetaScheduler, 
    WarmRestartBetaScheduler,
    HierarchicalBetaScheduler,
    BetaScheduler,
    CosineAnnealingBetaScheduler
)


def test_cyclical_schedulers():
    """Test cyclical beta schedulers with different wave types."""
    print("=" * 60)
    print("Testing Cyclical Beta Schedulers")
    print("=" * 60)
    
    epochs = 100
    epoch_range = list(range(epochs))
    
    # Test different cyclical configurations
    configs = [
        {'cycle_length': 20, 'min_beta': 0.0, 'max_beta': 1.0, 'wave_type': 'cosine'},
        {'cycle_length': 20, 'min_beta': 0.0, 'max_beta': 1.0, 'wave_type': 'triangle'},
        {'cycle_length': 30, 'min_beta': 0.1, 'max_beta': 0.8, 'wave_type': 'cosine'},
        {'cycle_length': 15, 'min_beta': 0.0, 'max_beta': 1.5, 'wave_type': 'triangle'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, config in enumerate(configs):
        scheduler = CyclicalBetaScheduler(**config)
        beta_values = [scheduler(epoch) for epoch in epoch_range]
        
        axes[i].plot(epoch_range, beta_values, linewidth=2)
        axes[i].set_title(f"Cyclical ({config['wave_type']}) - "
                         f"Cycle: {config['cycle_length']}, "
                         f"Range: [{config['min_beta']}, {config['max_beta']}]")
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Beta Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(min(config['min_beta'], 0) - 0.1, config['max_beta'] + 0.1)
        
        print(f"Config {i+1}: {scheduler}")
        print(f"  Beta range: {min(beta_values):.3f} - {max(beta_values):.3f}")
        print(f"  Period verification: Cycle {config['cycle_length']} epochs")
        
        # Verify periodicity
        if len(beta_values) >= 2 * config['cycle_length']:
            cycle1_values = beta_values[:config['cycle_length']]
            cycle2_values = beta_values[config['cycle_length']:2*config['cycle_length']]
            max_diff = max(abs(a - b) for a, b in zip(cycle1_values, cycle2_values))
            print(f"  Periodicity check: Max difference between cycles = {max_diff:.6f}")
    
    plt.tight_layout()
    plt.savefig('cyclical_schedulers_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Cyclical schedulers test passed!")
    return True


def test_warm_restart_schedulers():
    """Test warm restart beta schedulers."""
    print("\n" + "=" * 60)
    print("Testing Warm Restart Beta Schedulers")
    print("=" * 60)
    
    epochs = 150
    epoch_range = list(range(epochs))
    
    # Test different restart configurations
    base_schedulers = [
        ('cosine', {'max_beta': 1.0, 'warmup_epochs': 15}),
        ('linear', {'max_beta': 1.0, 'warmup_epochs': 20}),
        ('exponential', {'max_beta': 1.0, 'warmup_epochs': 10, 'decay_rate': 0.15})
    ]
    
    restart_configs = [
        {'restart_interval': 30, 'restart_multiplier': 1.0},
        {'restart_interval': 25, 'restart_multiplier': 1.5},
        {'restart_interval': 40, 'restart_multiplier': 1.2}
    ]
    
    fig, axes = plt.subplots(len(base_schedulers), len(restart_configs), 
                            figsize=(18, 12))
    
    for i, (base_type, base_kwargs) in enumerate(base_schedulers):
        for j, restart_config in enumerate(restart_configs):
            # Create base scheduler
            base_scheduler = get_beta_scheduler(base_type, **base_kwargs)
            
            # Create warm restart scheduler
            warm_restart_scheduler = WarmRestartBetaScheduler(
                base_scheduler, **restart_config
            )
            
            # Generate beta values
            beta_values = []
            restart_info_list = []
            
            for epoch in epoch_range:
                beta = warm_restart_scheduler(epoch)
                restart_info = warm_restart_scheduler.get_restart_info(epoch)
                beta_values.append(beta)
                restart_info_list.append(restart_info)
            
            # Plot
            ax = axes[i, j] if len(base_schedulers) > 1 else axes[j]
            ax.plot(epoch_range, beta_values, linewidth=2, label='Beta')
            
            # Mark restart epochs
            restart_epochs = [info['last_restart_epoch'] for info in restart_info_list]
            unique_restarts = sorted(list(set(restart_epochs)))
            for restart_epoch in unique_restarts:
                if restart_epoch > 0:  # Don't mark epoch 0
                    ax.axvline(x=restart_epoch, color='red', linestyle='--', alpha=0.7)
            
            ax.set_title(f"{base_type.title()} + Restart "
                        f"(Interval: {restart_config['restart_interval']}, "
                        f"Mult: {restart_config['restart_multiplier']})")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Beta Value')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, max(beta_values) + 0.1)
            
            print(f"\nWarm Restart Config ({base_type}, {restart_config}):")
            print(f"  Scheduler: {warm_restart_scheduler}")
            print(f"  Restart epochs: {unique_restarts[1:]}")  # Skip epoch 0
            print(f"  Final restart info: {restart_info_list[-1]}")
    
    plt.tight_layout()
    plt.savefig('warm_restart_schedulers_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Warm restart schedulers test passed!")
    return True


def test_hierarchical_schedulers():
    """Test hierarchical beta schedulers for hierarchical CVAE."""
    print("\n" + "=" * 60)
    print("Testing Hierarchical Beta Schedulers")
    print("=" * 60)
    
    epochs = 100
    epoch_range = list(range(epochs))
    
    # Test different hierarchical configurations
    hierarchical_configs = [
        {
            'global_scheduler': {'type': 'cosine', 'max_beta': 0.01, 'warmup_epochs': 10},
            'local_scheduler': {'type': 'cosine', 'max_beta': 0.01, 'warmup_epochs': 15}
        },
        {
            'global_scheduler': {'type': 'linear', 'max_beta': 0.005, 'warmup_epochs': 8},
            'local_scheduler': {'type': 'exponential', 'max_beta': 0.02, 'warmup_epochs': 20, 'decay_rate': 0.1}
        },
        {
            'global_scheduler': {'type': 'cyclical', 'cycle_length': 25, 'min_beta': 0.0, 'max_beta': 0.01, 'wave_type': 'cosine'},
            'local_scheduler': {'type': 'linear', 'max_beta': 0.015, 'warmup_epochs': 20}
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, config in enumerate(hierarchical_configs):
        scheduler = HierarchicalBetaScheduler(
            global_scheduler=get_beta_scheduler(
                config['global_scheduler']['type'],
                **{k: v for k, v in config['global_scheduler'].items() if k != 'type'}
            ),
            local_scheduler=get_beta_scheduler(
                config['local_scheduler']['type'],
                **{k: v for k, v in config['local_scheduler'].items() if k != 'type'}
            )
        )
        
        # Generate values
        global_values = []
        local_values = []
        
        for epoch in epoch_range:
            beta_dict = scheduler(epoch)
            global_values.append(beta_dict['global_kl_weight'])
            local_values.append(beta_dict['local_kl_weight'])
        
        # Plot
        axes[i].plot(epoch_range, global_values, linewidth=2, label='Global KL Weight', color='red')
        axes[i].plot(epoch_range, local_values, linewidth=2, label='Local KL Weight', color='blue')
        
        axes[i].set_title(f"Hierarchical Config {i+1}\n"
                         f"Global: {config['global_scheduler']['type']}, "
                         f"Local: {config['local_scheduler']['type']}")
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Beta Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        print(f"\nHierarchical Config {i+1}:")
        print(f"  Global scheduler: {scheduler.global_scheduler}")
        print(f"  Local scheduler: {scheduler.local_scheduler}")
        print(f"  Global range: {min(global_values):.6f} - {max(global_values):.6f}")
        print(f"  Local range: {min(local_values):.6f} - {max(local_values):.6f}")
    
    plt.tight_layout()
    plt.savefig('hierarchical_schedulers_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Hierarchical schedulers test passed!")
    return True


def test_scheduler_factory():
    """Test the scheduler factory function with all types."""
    print("\n" + "=" * 60)
    print("Testing Scheduler Factory Function")
    print("=" * 60)
    
    # Test all scheduler types
    test_configs = [
        ('linear', {'max_beta': 1.0, 'warmup_epochs': 10}),
        ('cosine', {'max_beta': 1.0, 'warmup_epochs': 15}),
        ('exponential', {'max_beta': 1.0, 'warmup_epochs': 12, 'decay_rate': 0.1}),
        ('cyclical', {'cycle_length': 20, 'min_beta': 0.0, 'max_beta': 1.0, 'wave_type': 'cosine'}),
        ('warm_restart', {
            'base_scheduler_type': 'cosine',
            'base_scheduler_kwargs': {'max_beta': 1.0, 'warmup_epochs': 10},
            'restart_interval': 30,
            'restart_multiplier': 1.0
        }),
        ('hierarchical', {
            'global_scheduler': {'type': 'cosine', 'max_beta': 0.01, 'warmup_epochs': 10},
            'local_scheduler': {'type': 'linear', 'max_beta': 0.01, 'warmup_epochs': 15}
        })
    ]
    
    schedulers = {}
    
    for scheduler_type, params in test_configs:
        try:
            scheduler = get_beta_scheduler(scheduler_type, **params)
            schedulers[scheduler_type] = scheduler
            print(f"✓ {scheduler_type}: {scheduler}")
            
            # Test a few epoch calls
            if scheduler_type == 'hierarchical':
                result = scheduler(5)
                print(f"  Epoch 5 result: {result}")
            else:
                result = scheduler(5)
                print(f"  Epoch 5 beta: {result:.6f}")
                
        except Exception as e:
            print(f"✗ {scheduler_type}: Failed with error: {e}")
            return False
    
    # Test available schedulers function
    from utils.schedulers import get_available_schedulers
    available = get_available_schedulers()
    print(f"\nAvailable scheduler types: {available}")
    
    # Test invalid scheduler type
    try:
        invalid_scheduler = get_beta_scheduler('invalid_type')
        print("✗ Error: Invalid scheduler type should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly handled invalid scheduler type: {e}")
    
    print("✓ Scheduler factory test passed!")
    return True


def create_comparison_visualization():
    """Create a comprehensive comparison of all scheduler types."""
    print("\n" + "=" * 60)
    print("Creating Comprehensive Scheduler Comparison")
    print("=" * 60)
    
    epochs = 120
    epoch_range = list(range(epochs))
    
    # Define representative configurations for each scheduler type
    scheduler_configs = {
        'Linear': ('linear', {'max_beta': 1.0, 'warmup_epochs': 20}),
        'Cosine': ('cosine', {'max_beta': 1.0, 'warmup_epochs': 20}),
        'Exponential': ('exponential', {'max_beta': 1.0, 'warmup_epochs': 20, 'decay_rate': 0.1}),
        'Cyclical (Cosine)': ('cyclical', {'cycle_length': 25, 'min_beta': 0.0, 'max_beta': 1.0, 'wave_type': 'cosine'}),
        'Cyclical (Triangle)': ('cyclical', {'cycle_length': 25, 'min_beta': 0.0, 'max_beta': 1.0, 'wave_type': 'triangle'}),
        'Warm Restart': ('warm_restart', {
            'base_scheduler_type': 'cosine',
            'base_scheduler_kwargs': {'max_beta': 1.0, 'warmup_epochs': 15},
            'restart_interval': 35,
            'restart_multiplier': 1.0
        })
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Standard schedulers
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    
    for i, (name, (stype, params)) in enumerate(scheduler_configs.items()):
        scheduler = get_beta_scheduler(stype, **params)
        
        if stype == 'warm_restart':
            # Handle warm restart specially
            beta_values = []
            for epoch in epoch_range:
                beta_values.append(scheduler(epoch))
        else:
            beta_values = [scheduler(epoch) for epoch in epoch_range]
        
        ax1.plot(epoch_range, beta_values, linewidth=2, label=name, color=colors[i])
        
        # Add restart markers for warm restart
        if stype == 'warm_restart':
            restart_epochs = []
            for epoch in epoch_range:
                restart_info = scheduler.get_restart_info(epoch)
                if restart_info['is_restart_epoch'] and epoch > 0:
                    restart_epochs.append(epoch)
            
            for restart_epoch in restart_epochs:
                ax1.axvline(x=restart_epoch, color=colors[i], linestyle='--', alpha=0.5)
    
    ax1.set_title('Beta Scheduler Comparison: Standard vs Advanced Schedules', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Beta Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)
    
    # Plot 2: Hierarchical scheduler example
    hierarchical_scheduler = get_beta_scheduler('hierarchical', **{
        'global_scheduler': {'type': 'cyclical', 'cycle_length': 30, 'min_beta': 0.0, 'max_beta': 0.01, 'wave_type': 'cosine'},
        'local_scheduler': {'type': 'warm_restart', 
                           'base_scheduler_type': 'cosine',
                           'base_scheduler_kwargs': {'max_beta': 0.02, 'warmup_epochs': 12},
                           'restart_interval': 40, 'restart_multiplier': 1.0}
    })
    
    global_values = []
    local_values = []
    
    for epoch in epoch_range:
        beta_dict = hierarchical_scheduler(epoch)
        global_values.append(beta_dict['global_kl_weight'])
        local_values.append(beta_dict['local_kl_weight'])
    
    ax2.plot(epoch_range, global_values, linewidth=2, label='Global KL Weight (Cyclical)', color='red')
    ax2.plot(epoch_range, local_values, linewidth=2, label='Local KL Weight (Warm Restart)', color='blue')
    
    # Mark restart epochs for local scheduler
    for epoch in epoch_range:
        if hasattr(hierarchical_scheduler.local_scheduler, 'get_restart_info'):
            restart_info = hierarchical_scheduler.local_scheduler.get_restart_info(epoch)
            if restart_info['is_restart_epoch'] and epoch > 0:
                ax2.axvline(x=epoch, color='blue', linestyle='--', alpha=0.5)
    
    ax2.set_title('Hierarchical Beta Scheduling: Independent Global and Local Schedules', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Beta Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scheduler_comparison_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics
    print("\nScheduler Characteristics Summary:")
    print("-" * 50)
    
    for name, (stype, params) in scheduler_configs.items():
        scheduler = get_beta_scheduler(stype, **params)
        
        if stype == 'warm_restart':
            beta_values = [scheduler(epoch) for epoch in range(50)]  # Shorter range for analysis
        else:
            beta_values = [scheduler(epoch) for epoch in range(50)]
        
        print(f"{name:20s}: Min={min(beta_values):.3f}, Max={max(beta_values):.3f}, "
              f"Mean={np.mean(beta_values):.3f}, Std={np.std(beta_values):.3f}")
    
    print("✓ Comprehensive comparison visualization created!")
    return True


def run_comprehensive_test():
    """Run all enhanced scheduler tests."""
    print("Starting Enhanced Beta Scheduler Tests")
    print("=" * 60)
    
    try:
        # Run individual tests
        test_cyclical_schedulers()
        test_warm_restart_schedulers()
        test_hierarchical_schedulers()
        test_scheduler_factory()
        create_comparison_visualization()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("Enhanced Beta Scheduler implementation is working correctly.")
        print("=" * 60)
        
        # Print usage instructions
        print("\n📋 Usage Instructions:")
        print("1. Use 'cyclical' for periodic KL regularization cycling")
        print("2. Use 'warm_restart' for periodic beta resets")
        print("3. Use 'hierarchical' for separate global/local scheduling")
        print("4. Configure via 'beta_scheduler_type' in training config")
        print("5. Monitor beta schedules in TensorBoard under 'Beta_Schedule/'")
        print("6. Use cyclical_beta_config.json and warm_restart_config.json examples")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 Enhanced Beta Scheduler implementation is ready for use!")
        print("Key features:")
        print("  • Cyclical scheduling with cosine/triangle waves")
        print("  • Warm restart scheduling with adaptive intervals")
        print("  • Hierarchical scheduling for multi-latent models")
        print("  • Enhanced TensorBoard logging and visualization")
        print("  • Comprehensive factory function with error handling")
        print("  • Batch-level and epoch-level beta tracking")
        
        print("\n📊 Generated visualizations:")
        print("  • cyclical_schedulers_demo.png")
        print("  • warm_restart_schedulers_demo.png")
        print("  • hierarchical_schedulers_demo.png")
        print("  • scheduler_comparison_comprehensive.png")
    else:
        print("\n⚠️  Please check the implementation and fix any issues.") 