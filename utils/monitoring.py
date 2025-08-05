#!/usr/bin/env python3
"""
Enhanced Monitoring and Logging for ABR Diffusion Training

Provides comprehensive monitoring capabilities:
- Detailed loss component tracking
- Gradient flow analysis
- Class-wise performance monitoring
- Model health diagnostics
- Early warning systems
- Interactive training dashboards
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import logging
import json
import time
from pathlib import Path
from collections import defaultdict, deque
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Comprehensive training monitor for ABR diffusion model.
    """
    
    def __init__(
        self,
        log_dir: str,
        n_classes: int = 5,
        max_history: int = 1000,
        save_frequency: int = 10
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_classes = n_classes
        self.max_history = max_history
        self.save_frequency = save_frequency
        
        # Initialize tracking variables
        self.reset_tracking()
        
        # Setup monitoring files
        self.setup_logging_files()
        
        logger.info(f"Training monitor initialized: {self.log_dir}")
    
    def reset_tracking(self):
        """Reset all tracking variables."""
        # Loss tracking
        self.loss_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.gradient_norms = deque(maxlen=self.max_history)
        self.learning_rates = deque(maxlen=self.max_history)
        
        # Performance tracking
        self.class_accuracies = defaultdict(lambda: deque(maxlen=self.max_history))
        self.class_losses = defaultdict(lambda: deque(maxlen=self.max_history))
        self.confusion_matrices = deque(maxlen=50)  # Store fewer confusion matrices
        
        # Model health tracking
        self.gradient_flow_issues = deque(maxlen=100)
        self.loss_spikes = deque(maxlen=100)
        self.convergence_metrics = deque(maxlen=self.max_history)
        
        # Timing tracking
        self.batch_times = deque(maxlen=200)
        self.epoch_times = deque(maxlen=100)
        
        # Current epoch tracking
        self.current_epoch = 0
        self.current_step = 0
    
    def setup_logging_files(self):
        """Setup detailed logging files."""
        # Detailed metrics log
        self.metrics_file = self.log_dir / "detailed_metrics.jsonl"
        
        # Loss component log
        self.loss_file = self.log_dir / "loss_components.jsonl"
        
        # Model health log
        self.health_file = self.log_dir / "model_health.jsonl"
        
        # Training summary
        self.summary_file = self.log_dir / "training_summary.json"
        
        # Create headers
        with open(self.metrics_file, 'w') as f:
            f.write("")  # Empty file
    
    def log_step(
        self,
        step: int,
        epoch: int,
        loss_components: Dict[str, float],
        outputs: Optional[Dict] = None,
        targets: Optional[Dict] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional = None
    ):
        """Log detailed information for a training step."""
        self.current_step = step
        self.current_epoch = epoch
        
        timestamp = time.time()
        
        # Log loss components
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.loss_history[key].append(value)
        
        # Log learning rate
        if optimizer is not None:
            lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(lr)
        
        # Analyze gradient flow
        if model is not None:
            grad_norm = self._compute_gradient_norm(model)
            self.gradient_norms.append(grad_norm)
            
            # Check for gradient flow issues
            self._check_gradient_health(model, step)
        
        # Analyze classification performance
        if outputs is not None and targets is not None:
            self._analyze_classification_performance(outputs, targets)
        
        # Check for loss spikes
        self._check_loss_health(loss_components, step)
        
        # Save detailed metrics every few steps
        if step % self.save_frequency == 0:
            self._save_step_metrics(step, epoch, loss_components, timestamp)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch_time: float
    ):
        """Log epoch-level information."""
        self.epoch_times.append(epoch_time)
        
        # Compute convergence metrics
        convergence_score = self._compute_convergence_score()
        self.convergence_metrics.append(convergence_score)
        
        # Log epoch summary
        epoch_summary = {
            'epoch': epoch,
            'timestamp': time.time(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'epoch_time': epoch_time,
            'convergence_score': convergence_score,
            'avg_gradient_norm': np.mean(list(self.gradient_norms)[-100:]) if self.gradient_norms else 0,
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0
        }
        
        # Save epoch summary
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(epoch_summary) + '\n')
        
        # Generate periodic reports
        if epoch % 10 == 0:
            self.generate_training_report()
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute the total gradient norm."""
        total_norm = 0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return np.sqrt(total_norm) if param_count > 0 else 0.0
    
    def _check_gradient_health(self, model: nn.Module, step: int):
        """Check for gradient flow issues."""
        zero_grad_count = 0
        very_small_grad_count = 0
        very_large_grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                
                if grad_norm == 0:
                    zero_grad_count += 1
                elif grad_norm < 1e-7:
                    very_small_grad_count += 1
                elif grad_norm > 100:
                    very_large_grad_count += 1
        
        total_params = sum(1 for _ in model.parameters())
        
        # Check for issues
        issues = []
        if zero_grad_count > total_params * 0.1:
            issues.append(f"Many parameters ({zero_grad_count}) have zero gradients")
        
        if very_small_grad_count > total_params * 0.3:
            issues.append(f"Many parameters ({very_small_grad_count}) have very small gradients")
        
        if very_large_grad_count > 0:
            issues.append(f"Some parameters ({very_large_grad_count}) have very large gradients")
        
        if issues:
            issue_record = {
                'step': step,
                'timestamp': time.time(),
                'issues': issues,
                'zero_grad_count': zero_grad_count,
                'small_grad_count': very_small_grad_count,
                'large_grad_count': very_large_grad_count,
                'total_params': total_params
            }
            self.gradient_flow_issues.append(issue_record)
    
    def _analyze_classification_performance(self, outputs: Dict, targets: Dict):
        """Analyze per-class classification performance."""
        if 'class' not in outputs or 'target' not in targets:
            return
        
        with torch.no_grad():
            class_logits = outputs['class']
            true_labels = targets['target']
            
            # Get predictions
            pred_labels = class_logits.argmax(dim=1)
            
            # Compute per-class accuracy
            for class_id in range(self.n_classes):
                class_mask = true_labels == class_id
                if class_mask.sum() > 0:
                    class_acc = (pred_labels[class_mask] == true_labels[class_mask]).float().mean().item()
                    self.class_accuracies[class_id].append(class_acc)
                    
                    # Compute per-class loss
                    class_loss = F.cross_entropy(
                        class_logits[class_mask], 
                        true_labels[class_mask]
                    ).item()
                    self.class_losses[class_id].append(class_loss)
    
    def _check_loss_health(self, loss_components: Dict, step: int):
        """Check for unusual loss behavior."""
        total_loss = loss_components.get('total_loss', 0)
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()
        
        # Check for loss spikes
        if len(self.loss_history['total_loss']) > 10:
            recent_losses = list(self.loss_history['total_loss'])[-10:]
            avg_recent = np.mean(recent_losses)
            
            if total_loss > avg_recent * 3:  # Loss spike
                spike_record = {
                    'step': step,
                    'timestamp': time.time(),
                    'current_loss': total_loss,
                    'recent_avg': avg_recent,
                    'spike_ratio': total_loss / avg_recent
                }
                self.loss_spikes.append(spike_record)
    
    def _compute_convergence_score(self) -> float:
        """Compute a convergence score based on loss stability."""
        if len(self.loss_history['total_loss']) < 20:
            return 0.0
        
        recent_losses = list(self.loss_history['total_loss'])[-20:]
        
        # Compute trend (negative slope is good)
        x = np.arange(len(recent_losses))
        slope = np.polyfit(x, recent_losses, 1)[0]
        
        # Compute stability (low variance is good)
        variance = np.var(recent_losses)
        
        # Combine into score (higher is better)
        convergence_score = max(0, -slope) / (1 + variance)
        return convergence_score
    
    def _save_step_metrics(self, step: int, epoch: int, loss_components: Dict, timestamp: float):
        """Save detailed step metrics."""
        step_record = {
            'step': step,
            'epoch': epoch,
            'timestamp': timestamp,
            'loss_components': {k: v.item() if isinstance(v, torch.Tensor) else v 
                              for k, v in loss_components.items()},
            'gradient_norm': self.gradient_norms[-1] if self.gradient_norms else 0,
            'learning_rate': self.learning_rates[-1] if self.learning_rates else 0
        }
        
        with open(self.loss_file, 'a') as f:
            f.write(json.dumps(step_record) + '\n')
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        report = {
            'timestamp': time.time(),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_training_time': sum(self.epoch_times),
            
            # Loss statistics
            'loss_statistics': self._compute_loss_statistics(),
            
            # Gradient health
            'gradient_health': self._compute_gradient_health_summary(),
            
            # Classification performance
            'classification_performance': self._compute_classification_summary(),
            
            # Model health
            'model_health': {
                'gradient_issues': len(self.gradient_flow_issues),
                'loss_spikes': len(self.loss_spikes),
                'convergence_score': self.convergence_metrics[-1] if self.convergence_metrics else 0
            },
            
            # Training efficiency
            'efficiency': {
                'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
                'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0
            }
        }
        
        # Save training report
        with open(self.summary_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved: {self.summary_file}")
    
    def _compute_loss_statistics(self) -> Dict:
        """Compute loss statistics."""
        stats = {}
        
        for loss_name, values in self.loss_history.items():
            if values:
                stats[loss_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': 'decreasing' if len(values) > 10 and values[-1] < values[-10] else 'stable'
                }
        
        return stats
    
    def _compute_gradient_health_summary(self) -> Dict:
        """Compute gradient health summary."""
        return {
            'avg_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0,
            'gradient_stability': np.std(self.gradient_norms) if len(self.gradient_norms) > 1 else 0,
            'recent_gradient_issues': len([x for x in self.gradient_flow_issues if x['step'] > self.current_step - 100])
        }
    
    def _compute_classification_summary(self) -> Dict:
        """Compute classification performance summary."""
        summary = {}
        
        for class_id in range(self.n_classes):
            if self.class_accuracies[class_id]:
                summary[f'class_{class_id}'] = {
                    'accuracy': np.mean(list(self.class_accuracies[class_id])[-20:]),
                    'loss': np.mean(list(self.class_losses[class_id])[-20:]),
                    'sample_count': len(self.class_accuracies[class_id])
                }
        
        return summary
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Generate comprehensive training progress plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].set_title('Loss Components')
        for loss_name, values in self.loss_history.items():
            if values and loss_name != 'total_loss':
                axes[0, 0].plot(values, label=loss_name, alpha=0.7)
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Total loss
        axes[0, 1].set_title('Total Loss')
        if self.loss_history['total_loss']:
            axes[0, 1].plot(self.loss_history['total_loss'])
        axes[0, 1].set_yscale('log')
        
        # Gradient norms
        axes[0, 2].set_title('Gradient Norms')
        if self.gradient_norms:
            axes[0, 2].plot(self.gradient_norms)
        axes[0, 2].set_yscale('log')
        
        # Learning rate
        axes[1, 0].set_title('Learning Rate')
        if self.learning_rates:
            axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_yscale('log')
        
        # Class accuracies
        axes[1, 1].set_title('Class Accuracies')
        for class_id in range(self.n_classes):
            if self.class_accuracies[class_id]:
                axes[1, 1].plot(self.class_accuracies[class_id], label=f'Class {class_id}')
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        # Convergence score
        axes[1, 2].set_title('Convergence Score')
        if self.convergence_metrics:
            axes[1, 2].plot(self.convergence_metrics)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.log_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def get_early_stopping_recommendation(self) -> Dict:
        """Provide early stopping recommendation based on monitoring data."""
        if len(self.loss_history['total_loss']) < 50:
            return {'should_stop': False, 'reason': 'Insufficient training data'}
        
        recent_losses = list(self.loss_history['total_loss'])[-30:]
        
        # Check for stagnation
        loss_variance = np.var(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        recommendations = []
        
        if loss_variance < 1e-6 and abs(loss_trend) < 1e-6:
            recommendations.append("Loss has stagnated")
        
        if len(self.loss_spikes) > 5:
            recommendations.append("Too many loss spikes detected")
        
        if len(self.gradient_flow_issues) > 10:
            recommendations.append("Persistent gradient flow issues")
        
        convergence_score = self.convergence_metrics[-1] if self.convergence_metrics else 0
        if convergence_score < 0.01:
            recommendations.append("Poor convergence score")
        
        return {
            'should_stop': len(recommendations) >= 2,
            'reasons': recommendations,
            'convergence_score': convergence_score,
            'loss_trend': loss_trend,
            'recent_variance': loss_variance
        }


def create_training_monitor(config, log_dir: str) -> TrainingMonitor:
    """Create and configure training monitor."""
    monitor = TrainingMonitor(
        log_dir=log_dir,
        n_classes=config.data.n_classes,
        max_history=1000,
        save_frequency=10
    )
    
    logger.info(f"Training monitor created: {log_dir}")
    return monitor