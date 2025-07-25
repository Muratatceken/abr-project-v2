"""
Training Visualization Module
Comprehensive plotting and visualization for ABR training progress.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime


class TrainingVisualizer:
    """Comprehensive training visualization and plotting."""
    
    def __init__(self, plot_dir: str = "plots", experiment_name: str = "abr_training"):
        """Initialize the visualizer."""
        self.plot_dir = Path(plot_dir)
        self.experiment_name = experiment_name
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics history
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'learning_rate': [],
            'train_metrics': {},
            'val_metrics': {}
        }
        
        # Set style - use compatible matplotlib style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        sns.set_palette("husl")
        
    def update_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                      learning_rate: float):
        """Update metrics history for plotting."""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_metrics.get('total_loss', 0))
        
        # Extract validation loss and F1 with fallbacks
        val_loss = val_metrics.get('gen_total_loss', val_metrics.get('direct_total_loss', 0))
        val_f1 = val_metrics.get('gen_f1_macro', val_metrics.get('direct_f1_macro', 0))
        
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_f1'].append(val_f1)
        self.metrics_history['learning_rate'].append(learning_rate)
        
        # Store detailed metrics
        for key, value in train_metrics.items():
            if key not in self.metrics_history['train_metrics']:
                self.metrics_history['train_metrics'][key] = []
            self.metrics_history['train_metrics'][key].append(value)
            
        for key, value in val_metrics.items():
            if key not in self.metrics_history['val_metrics']:
                self.metrics_history['val_metrics'][key] = []
            self.metrics_history['val_metrics'][key].append(value)
    
    def plot_training_curves(self, save: bool = True) -> Optional[str]:
        """Plot main training curves (loss and F1)."""
        if len(self.metrics_history['epoch']) < 2:
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        epochs = self.metrics_history['epoch']
        
        # Training and validation loss
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation F1 Score
        ax2.plot(epochs, self.metrics_history['val_f1'], 'g-', label='Val F1', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Validation F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning Rate
        ax3.semilogy(epochs, self.metrics_history['learning_rate'], 'm-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate (log scale)')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        
        # Loss comparison (zoomed)
        if len(epochs) > 10:
            recent_epochs = epochs[-min(20, len(epochs)):]
            recent_train = self.metrics_history['train_loss'][-len(recent_epochs):]
            recent_val = self.metrics_history['val_loss'][-len(recent_epochs):]
            
            ax4.plot(recent_epochs, recent_train, 'b-', label='Train Loss', linewidth=2)
            ax4.plot(recent_epochs, recent_val, 'r-', label='Val Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Recent Loss (Last 20 Epochs)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.experiment_name}_training_curves.png"
            filepath = self.plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
    
    def plot_loss_components(self, save: bool = True) -> Optional[str]:
        """Plot individual loss components."""
        if len(self.metrics_history['epoch']) < 2:
            return None
            
        # Find loss components
        loss_components = [key for key in self.metrics_history['train_metrics'].keys() 
                          if 'loss' in key.lower() and key != 'total_loss']
        
        if not loss_components:
            return None
            
        n_components = len(loss_components)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_components == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        epochs = self.metrics_history['epoch']
        
        for i, component in enumerate(loss_components):
            if i < len(axes):
                train_values = self.metrics_history['train_metrics'].get(component, [])
                val_values = self.metrics_history['val_metrics'].get(f'direct_{component}', [])
                
                if train_values:
                    axes[i].plot(epochs, train_values, 'b-', label=f'Train {component}', linewidth=2)
                if val_values:
                    axes[i].plot(epochs, val_values, 'r-', label=f'Val {component}', linewidth=2)
                
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')
                axes[i].set_title(f'{component.replace("_", " ").title()}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_components, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.experiment_name}_loss_components.png"
            filepath = self.plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
    
    def plot_validation_metrics(self, save: bool = True) -> Optional[str]:
        """Plot validation metrics over time."""
        if len(self.metrics_history['epoch']) < 2:
            return None
            
        # Find validation metrics (excluding loss components)
        val_metrics = [key for key in self.metrics_history['val_metrics'].keys() 
                      if not key.endswith('_loss') and 'f1' in key.lower()]
        
        if not val_metrics:
            return None
            
        n_metrics = len(val_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        epochs = self.metrics_history['epoch']
        colors = plt.cm.Set1(np.linspace(0, 1, n_metrics))
        
        for i, metric in enumerate(val_metrics):
            if i < len(axes):
                values = self.metrics_history['val_metrics'].get(metric, [])
                if values:
                    axes[i].plot(epochs, values, color=colors[i], linewidth=2, marker='o', markersize=4)
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel('Score')
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.experiment_name}_validation_metrics.png"
            filepath = self.plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
    
    def plot_training_summary(self, save: bool = True) -> Optional[str]:
        """Create a comprehensive training summary plot."""
        if len(self.metrics_history['epoch']) < 2:
            return None
            
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        epochs = self.metrics_history['epoch']
        
        # 1. Main loss curves (top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress: Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. F1 Score (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(epochs, self.metrics_history['val_f1'], 'g-', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Validation F1', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Learning Rate (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.semilogy(epochs, self.metrics_history['learning_rate'], 'm-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('LR Schedule', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Recent performance (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        if len(epochs) > 10:
            recent_epochs = epochs[-min(20, len(epochs)):]
            recent_train = self.metrics_history['train_loss'][-len(recent_epochs):]
            recent_val = self.metrics_history['val_loss'][-len(recent_epochs):]
            
            ax4.plot(recent_epochs, recent_train, 'b-', label='Train', linewidth=2)
            ax4.plot(recent_epochs, recent_val, 'r-', label='Val', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Recent Performance', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Training statistics (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate statistics
        current_epoch = epochs[-1] if epochs else 0
        best_val_loss = min(self.metrics_history['val_loss']) if self.metrics_history['val_loss'] else 0
        best_val_f1 = max(self.metrics_history['val_f1']) if self.metrics_history['val_f1'] else 0
        current_lr = self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else 0
        
        stats_text = f"""Training Statistics:
        
Current Epoch: {current_epoch}
Best Val Loss: {best_val_loss:.4f}
Best Val F1: {best_val_f1:.4f}
Current LR: {current_lr:.2e}

Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax5.set_title('Statistics', fontweight='bold')
        
        # 6. Loss components (bottom row)
        loss_components = [key for key in self.metrics_history['train_metrics'].keys() 
                          if 'loss' in key.lower() and key != 'total_loss']
        
        if loss_components:
            # Show top 3 loss components
            top_components = loss_components[:3]
            for i, component in enumerate(top_components):
                ax = fig.add_subplot(gs[2, i])
                train_values = self.metrics_history['train_metrics'].get(component, [])
                if train_values:
                    ax.plot(epochs, train_values, linewidth=2)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'{component.replace("_", " ").title()}', fontweight='bold')
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'ABR Training Summary: {self.experiment_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            filename = f"{self.experiment_name}_training_summary.png"
            filepath = self.plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
    
    def save_metrics_csv(self) -> str:
        """Save metrics history to CSV file."""
        # Prepare data for CSV
        data = {
            'epoch': self.metrics_history['epoch'],
            'train_loss': self.metrics_history['train_loss'],
            'val_loss': self.metrics_history['val_loss'],
            'val_f1': self.metrics_history['val_f1'],
            'learning_rate': self.metrics_history['learning_rate']
        }
        
        # Add other metrics
        for key, values in self.metrics_history['train_metrics'].items():
            data[f'train_{key}'] = values + [None] * (len(data['epoch']) - len(values))
            
        for key, values in self.metrics_history['val_metrics'].items():
            data[f'val_{key}'] = values + [None] * (len(data['epoch']) - len(values))
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filename = f"{self.experiment_name}_metrics.csv"
        filepath = self.plot_dir / filename
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def create_all_plots(self, save: bool = True) -> List[str]:
        """Create all available plots."""
        created_plots = []
        
        # Main training curves
        plot_path = self.plot_training_curves(save)
        if plot_path:
            created_plots.append(plot_path)
        
        # Loss components
        plot_path = self.plot_loss_components(save)
        if plot_path:
            created_plots.append(plot_path)
        
        # Validation metrics
        plot_path = self.plot_validation_metrics(save)
        if plot_path:
            created_plots.append(plot_path)
        
        # Training summary
        plot_path = self.plot_training_summary(save)
        if plot_path:
            created_plots.append(plot_path)
        
        # Save metrics CSV
        csv_path = self.save_metrics_csv()
        created_plots.append(csv_path)
        
        return created_plots 