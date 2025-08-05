#!/usr/bin/env python3
"""
Detailed Statistical Analysis of Training Results
Advanced metrics and insights for ABR model training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import re
from pathlib import Path

def advanced_convergence_analysis(log_path: str):
    """Perform advanced statistical analysis of convergence patterns."""
    
    # Parse training data
    step_data = []
    epoch_data = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_epoch = 0
    for line in lines:
        # Parse step data
        step_match = re.search(r'Step (\d+) - Total: ([\d.]+), Signal: ([\d.]+), Class: ([\d.]+), Peak: ([\d.]+), PeakLat: ([\d.]+), PeakAmp: ([\d.]+), Thresh: ([\d.]+)', line)
        if step_match:
            step_data.append({
                'step': int(step_match.group(1)),
                'epoch': current_epoch,
                'total_loss': float(step_match.group(2)),
                'signal_loss': float(step_match.group(3)),
                'class_loss': float(step_match.group(4)),
                'peak_loss': float(step_match.group(5)),
                'peak_lat_loss': float(step_match.group(6)),
                'peak_amp_loss': float(step_match.group(7)),
                'thresh_loss': float(step_match.group(8)),
            })
        
        # Track epoch changes
        epoch_match = re.search(r'Epoch (\d+)/\d+ Summary:', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # Parse epoch summaries
        train_match = re.search(r'Train - Total: ([\d.]+), Signal: ([\d.]+), Class: ([\d.]+), Peak: ([\d.]+), Thresh: ([\d.]+)', line)
        if train_match:
            epoch_data.append({
                'epoch': current_epoch,
                'type': 'train',
                'total_loss': float(train_match.group(1)),
                'signal_loss': float(train_match.group(2)),
                'class_loss': float(train_match.group(3)),
                'peak_loss': float(train_match.group(4)),
                'thresh_loss': float(train_match.group(5)),
            })
    
    df_steps = pd.DataFrame(step_data)
    df_epochs = pd.DataFrame(epoch_data)
    
    print("🔬 ADVANCED STATISTICAL ANALYSIS")
    print("="*60)
    
    # 1. Loss Distribution Analysis
    print("\n📊 LOSS DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    for component in ['total_loss', 'signal_loss', 'class_loss', 'peak_loss', 'thresh_loss']:
        if component in df_steps.columns:
            values = df_steps[component].values
            
            # Remove extreme outliers for better analysis
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
            
            print(f"\n{component.replace('_', ' ').title()}:")
            print(f"  Mean: {np.mean(filtered_values):.3f}")
            print(f"  Median: {np.median(filtered_values):.3f}")
            print(f"  Std: {np.std(filtered_values):.3f}")
            print(f"  Skewness: {stats.skew(filtered_values):.3f}")
            print(f"  Kurtosis: {stats.kurtosis(filtered_values):.3f}")
            print(f"  CV: {np.std(filtered_values)/np.mean(filtered_values)*100:.1f}%")
    
    # 2. Convergence Rate Analysis
    print("\n🚀 CONVERGENCE RATE ANALYSIS:")
    print("-" * 40)
    
    # Fit exponential decay models
    total_losses = df_steps['total_loss'].values
    steps = df_steps['step'].values
    
    # Log-transform for exponential fitting
    log_losses = np.log(total_losses + 1e-8)  # Add small constant to avoid log(0)
    
    # Fit polynomial to log-losses (exponential decay)
    try:
        coeffs = np.polyfit(steps, log_losses, 1)
        decay_rate = -coeffs[0]
        r_squared = stats.pearsonr(steps, log_losses)[0]**2
        
        print(f"Exponential Decay Rate: {decay_rate:.6f} per step")
        print(f"Half-life: {np.log(2)/decay_rate:.1f} steps")
        print(f"R²: {r_squared:.3f}")
        
        # Time to 1% of initial loss
        initial_loss = total_losses[0]
        target_loss = initial_loss * 0.01
        steps_to_1pct = np.log(target_loss/initial_loss) / (-decay_rate)
        print(f"Steps to 1% of initial loss: {steps_to_1pct:.0f}")
        
    except Exception as e:
        print(f"Could not fit exponential model: {e}")
    
    # 3. Learning Phase Detection
    print("\n🔍 LEARNING PHASE DETECTION:")
    print("-" * 40)
    
    # Detect different learning phases using change point detection
    losses = df_steps['total_loss'].values
    
    # Smooth the losses for better change point detection
    window_size = min(10, len(losses) // 4)
    if window_size > 1:
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        
        # Find significant changes in slope
        gradients = np.gradient(smoothed)
        gradient_changes = np.abs(np.gradient(gradients))
        
        # Find peaks in gradient changes (change points)
        peaks, _ = find_peaks(gradient_changes, height=np.percentile(gradient_changes, 80))
        
        print(f"Detected {len(peaks)} learning phase transitions:")
        for i, peak in enumerate(peaks):
            step_num = peak + window_size // 2  # Adjust for smoothing offset
            if step_num < len(losses):
                print(f"  Phase {i+1}: Step {step_num}, Loss {losses[step_num]:.2f}")
    
    # 4. Loss Component Correlation Analysis
    print("\n🔗 LOSS COMPONENT CORRELATION:")
    print("-" * 40)
    
    # Calculate correlation matrix
    components = ['signal_loss', 'class_loss', 'peak_loss', 'thresh_loss']
    correlation_data = df_steps[components].corr()
    
    print("Correlation Matrix:")
    for i, comp1 in enumerate(components):
        for j, comp2 in enumerate(components):
            if i < j:  # Only upper triangle
                corr = correlation_data.loc[comp1, comp2]
                print(f"  {comp1} vs {comp2}: {corr:.3f}")
    
    # 5. Outlier Analysis
    print("\n⚠️  OUTLIER ANALYSIS:")
    print("-" * 40)
    
    for component in components:
        values = df_steps[component].values
        
        # Z-score method
        z_scores = np.abs(stats.zscore(values))
        outliers_z = np.sum(z_scores > 3)
        
        # IQR method
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        outliers_iqr = np.sum((values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr))
        
        print(f"{component}: {outliers_z} Z-score outliers, {outliers_iqr} IQR outliers")
    
    # 6. Stability Analysis
    print("\n📈 STABILITY ANALYSIS:")
    print("-" * 40)
    
    # Calculate rolling statistics
    window = 20  # 20-step rolling window
    
    for component in ['total_loss'] + components:
        if len(df_steps) > window:
            rolling_mean = df_steps[component].rolling(window=window).mean()
            rolling_std = df_steps[component].rolling(window=window).std()
            rolling_cv = (rolling_std / rolling_mean * 100).dropna()
            
            # Recent stability (last 20% of training)
            recent_start = int(len(rolling_cv) * 0.8)
            recent_stability = rolling_cv.iloc[recent_start:].mean()
            
            print(f"{component}: Recent stability (CV): {recent_stability:.2f}%")
    
    return df_steps, df_epochs

def create_advanced_visualizations(df_steps: pd.DataFrame, output_dir: str = "advanced_analysis"):
    """Create advanced statistical visualizations."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # 1. Loss Distribution Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Loss Distribution Analysis', fontsize=16, fontweight='bold')
    
    components = ['total_loss', 'signal_loss', 'class_loss', 'peak_loss', 'thresh_loss']
    
    for i, component in enumerate(components):
        if i < 6:  # Only plot first 6 components
            row, col = i // 3, i % 3
            
            # Remove extreme outliers for visualization
            values = df_steps[component].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            filtered_values = values[(values >= q1 - 1.5*iqr) & (values <= q3 + 1.5*iqr)]
            
            # Histogram with KDE
            axes[row, col].hist(filtered_values, bins=30, alpha=0.7, density=True, color='skyblue')
            
            # Add KDE if we have enough data
            if len(filtered_values) > 10:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(filtered_values)
                x_range = np.linspace(filtered_values.min(), filtered_values.max(), 100)
                axes[row, col].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            axes[row, col].set_title(f'{component.replace("_", " ").title()} Distribution')
            axes[row, col].set_xlabel('Loss Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(filtered_values)
            std_val = np.std(filtered_values)
            axes[row, col].axvline(mean_val, color='red', linestyle='--', alpha=0.7)
            axes[row, col].text(0.05, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                              transform=axes[row, col].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplot
    if len(components) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_path / 'loss_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_components = ['signal_loss', 'class_loss', 'peak_loss', 'thresh_loss', 'peak_amp_loss']
    correlation_matrix = df_steps[corr_components].corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', ax=ax)
    ax.set_title('Loss Component Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Learning Phase Analysis
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Loss progression with phases
    steps = df_steps['step'].values
    total_losses = df_steps['total_loss'].values
    
    axes[0].plot(steps, total_losses, 'b-', alpha=0.7, linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_title('Loss Progression with Learning Phases')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Total Loss (log scale)')
    axes[0].grid(True, alpha=0.3)
    
    # Gradient analysis
    gradients = np.gradient(total_losses)
    axes[1].plot(steps[1:], gradients[1:], 'g-', alpha=0.7, linewidth=1)
    axes[1].set_title('Loss Gradient (Learning Rate)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss Gradient')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path / 'learning_phases.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Advanced visualizations saved to {output_path}")

def main():
    """Run advanced statistical analysis."""
    log_path = "logs/max_performance/training.log"
    
    if not Path(log_path).exists():
        print(f"❌ Training log not found: {log_path}")
        return
    
    print("🔬 STARTING ADVANCED STATISTICAL ANALYSIS")
    print("="*60)
    
    df_steps, df_epochs = advanced_convergence_analysis(log_path)
    create_advanced_visualizations(df_steps)
    
    print("\n🎯 ADVANCED ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()