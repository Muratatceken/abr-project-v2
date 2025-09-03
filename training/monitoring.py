"""
Comprehensive training monitoring framework for ABR transformer models.

This module implements real-time monitoring and analysis including:
- Training metrics and system resource monitoring
- Gradient flow analysis and dead neuron detection
- Model health indicators and bottleneck detection
- Real-time dashboards and alerting systems
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from collections import defaultdict, deque
from pathlib import Path
import json
import threading
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a specific time point."""
    
    timestamp: float
    epoch: int
    step: int
    metrics: Dict[str, float]
    system_metrics: Optional[Dict[str, float]] = None
    model_metrics: Optional[Dict[str, float]] = None


@dataclass
class AlertConfig:
    """Configuration for monitoring alerts."""
    
    metric_name: str
    threshold: float
    condition: str = "greater"  # "greater", "less", "equal"
    consecutive_violations: int = 3
    enabled: bool = True


class BaseMonitor(ABC):
    """
    Abstract base class for monitoring components.
    """
    
    @abstractmethod
    def update(self, **kwargs):
        """Update monitor with new data."""
        pass
    
    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of monitored data."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset monitor state."""
        pass


class MetricTracker(BaseMonitor):
    """
    Track and store training and validation metrics over time.
    
    Supports metric smoothing, trend analysis, and anomaly detection.
    """
    
    def __init__(
        self,
        smoothing_window: int = 10,
        max_history: int = 10000,
        enable_anomaly_detection: bool = True
    ):
        """
        Initialize metric tracker.
        
        Args:
            smoothing_window: Window size for metric smoothing
            max_history: Maximum number of metric snapshots to keep
            enable_anomaly_detection: Whether to enable anomaly detection
        """
        self.smoothing_window = smoothing_window
        self.max_history = max_history
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Storage
        self.snapshots = deque(maxlen=max_history)
        self.metrics_history = defaultdict(list)
        self.smoothed_metrics = defaultdict(list)
        
        # Anomaly detection
        self.anomaly_thresholds = {}
        self.anomaly_counts = defaultdict(int)
        
    def update(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        system_metrics: Optional[Dict[str, float]] = None,
        model_metrics: Optional[Dict[str, float]] = None
    ):
        """Update tracker with new metrics."""
        timestamp = time.time()
        
        # Create snapshot
        snapshot = MetricSnapshot(
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            metrics=metrics,
            system_metrics=system_metrics,
            model_metrics=model_metrics
        )
        
        self.snapshots.append(snapshot)
        
        # Update metrics history
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
            
            # Compute smoothed value
            recent_values = self.metrics_history[metric_name][-self.smoothing_window:]
            smoothed_value = np.mean(recent_values)
            self.smoothed_metrics[metric_name].append(smoothed_value)
            
            # Anomaly detection
            if self.enable_anomaly_detection:
                self._check_anomaly(metric_name, value)
                
    def _check_anomaly(self, metric_name: str, value: float):
        """Check if metric value is anomalous."""
        if metric_name not in self.anomaly_thresholds:
            # Initialize threshold based on historical data
            if len(self.metrics_history[metric_name]) > 20:
                values = np.array(self.metrics_history[metric_name])
                mean_val = np.mean(values)
                std_val = np.std(values)
                self.anomaly_thresholds[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'upper': mean_val + 3 * std_val,
                    'lower': mean_val - 3 * std_val
                }
        else:
            # Check for anomaly
            threshold = self.anomaly_thresholds[metric_name]
            if value > threshold['upper'] or value < threshold['lower']:
                self.anomaly_counts[metric_name] += 1
                if self.anomaly_counts[metric_name] >= 3:  # Consecutive anomalies
                    logger.warning(f"Anomaly detected in {metric_name}: {value:.4f} "
                                 f"(expected range: {threshold['lower']:.4f} - {threshold['upper']:.4f})")
            else:
                self.anomaly_counts[metric_name] = 0
                
    def get_metric_trend(self, metric_name: str, window: int = 50) -> Dict[str, float]:
        """
        Get trend analysis for a specific metric.
        
        Args:
            metric_name: Name of the metric
            window: Window size for trend analysis
            
        Returns:
            Dictionary with trend statistics
        """
        if metric_name not in self.metrics_history:
            return {}
            
        values = self.metrics_history[metric_name][-window:]
        if len(values) < 2:
            return {}
            
        # Linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Statistics
        recent_mean = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        overall_mean = np.mean(values)
        
        return {
            'slope': slope,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'recent_mean': recent_mean,
            'overall_mean': overall_mean,
            'improvement': recent_mean - overall_mean,
            'volatility': np.std(values),
            'n_samples': len(values)
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        summary = {
            'total_snapshots': len(self.snapshots),
            'metrics_tracked': list(self.metrics_history.keys()),
            'latest_metrics': {},
            'trends': {},
            'anomalies': dict(self.anomaly_counts)
        }
        
        if self.snapshots:
            latest = self.snapshots[-1]
            summary['latest_metrics'] = latest.metrics
            summary['latest_epoch'] = latest.epoch
            summary['latest_step'] = latest.step
            
        # Trend analysis for each metric
        for metric_name in self.metrics_history.keys():
            summary['trends'][metric_name] = self.get_metric_trend(metric_name)
            
        return summary
        
    def reset(self):
        """Reset tracker state."""
        self.snapshots.clear()
        self.metrics_history.clear()
        self.smoothed_metrics.clear()
        self.anomaly_thresholds.clear()
        self.anomaly_counts.clear()


class ResourceMonitor(BaseMonitor):
    """
    Monitor system resources during training.
    
    Tracks GPU/CPU usage, memory consumption, disk I/O, and training speed.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.resource_history = defaultdict(list)
        self.monitoring = False
        self.monitor_thread = None
        
        # Check available resources
        self.has_gpu = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                resources = self._collect_resources()
                timestamp = time.time()
                
                for resource_name, value in resources.items():
                    self.resource_history[resource_name].append((timestamp, value))
                    
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                
    def _collect_resources(self) -> Dict[str, float]:
        """Collect current resource usage."""
        resources = {}
        
        # CPU metrics
        resources['cpu_percent'] = psutil.cpu_percent()
        resources['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        resources['memory_percent'] = memory.percent
        resources['memory_available_gb'] = memory.available / (1024**3)
        resources['memory_used_gb'] = memory.used / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        resources['disk_percent'] = disk.percent
        resources['disk_free_gb'] = disk.free / (1024**3)
        
        # GPU metrics
        if self.has_gpu:
            for gpu_id in range(self.gpu_count):
                try:
                    # Memory usage
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    
                    resources[f'gpu_{gpu_id}_memory_allocated_gb'] = memory_allocated
                    resources[f'gpu_{gpu_id}_memory_reserved_gb'] = memory_reserved
                    
                    # GPU utilization (if nvidia-ml-py available)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        resources[f'gpu_{gpu_id}_utilization'] = util.gpu
                    except ImportError:
                        pass  # nvidia-ml-py not available
                    except Exception:
                        pass  # GPU monitoring failed
                        
                except Exception as e:
                    logger.debug(f"Failed to get GPU {gpu_id} metrics: {e}")
                    
        return resources
        
    def update(self, **kwargs):
        """Manual update (for compatibility with BaseMonitor)."""
        resources = self._collect_resources()
        timestamp = time.time()
        
        for resource_name, value in resources.items():
            self.resource_history[resource_name].append((timestamp, value))
            
    def get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage."""
        return self._collect_resources()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        summary = {
            'monitoring_active': self.monitoring,
            'has_gpu': self.has_gpu,
            'gpu_count': self.gpu_count,
            'current_resources': self.get_current_resources(),
            'resource_stats': {}
        }
        
        # Compute statistics for each resource
        for resource_name, history in self.resource_history.items():
            if not history:
                continue
                
            values = [value for _, value in history]
            summary['resource_stats'][resource_name] = {
                'mean': np.mean(values),
                'max': np.max(values),
                'min': np.min(values),
                'std': np.std(values),
                'current': values[-1] if values else None,
                'n_samples': len(values)
            }
            
        return summary
        
    def reset(self):
        """Reset resource monitoring history."""
        self.resource_history.clear()


class ModelHealthMonitor(BaseMonitor):
    """
    Monitor model health indicators during training.
    
    Tracks gradient norms, weight distributions, activation statistics,
    and detects potential training issues.
    """
    
    def __init__(self, track_activations: bool = False):
        """
        Initialize model health monitor.
        
        Args:
            track_activations: Whether to track activation statistics (expensive)
        """
        self.track_activations = track_activations
        
        # Storage
        self.gradient_history = defaultdict(list)
        self.weight_history = defaultdict(list)
        self.activation_history = defaultdict(list) if track_activations else None
        
        # Health indicators
        self.dead_neurons = {}
        self.gradient_issues = {}
        
        # Hooks for activation tracking
        self.activation_hooks = []
        
    def update(
        self,
        model: nn.Module,
        epoch: int = 0,
        step: int = 0,
        **kwargs
    ):
        """Update model health metrics."""
        # Gradient analysis
        self._analyze_gradients(model, epoch, step)
        
        # Weight analysis
        self._analyze_weights(model, epoch, step)
        
        # Activation analysis (if enabled)
        if self.track_activations:
            self._setup_activation_hooks(model)
            
    def _analyze_gradients(self, model: nn.Module, epoch: int, step: int):
        """Analyze gradient flow and detect issues."""
        total_norm = 0.0
        param_count = 0
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Layer-wise gradient norms
                layer_name = '.'.join(name.split('.')[:-1])  # Remove parameter name
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param_norm)
                
                # Check for gradient issues
                if param_norm > 10.0:  # Large gradient
                    if name not in self.gradient_issues:
                        self.gradient_issues[name] = []
                    self.gradient_issues[name].append({
                        'epoch': epoch,
                        'step': step,
                        'issue': 'large_gradient',
                        'value': param_norm
                    })
                elif param_norm < 1e-7:  # Very small gradient
                    if name not in self.gradient_issues:
                        self.gradient_issues[name] = []
                    self.gradient_issues[name].append({
                        'epoch': epoch,
                        'step': step,
                        'issue': 'small_gradient',
                        'value': param_norm
                    })
                    
        # Overall gradient norm
        total_norm = total_norm ** 0.5
        self.gradient_history['total_norm'].append((epoch, step, total_norm))
        
        # Layer-wise norms
        for layer_name, norms in layer_norms.items():
            layer_norm = np.sqrt(sum(norm ** 2 for norm in norms))
            self.gradient_history[f'{layer_name}_norm'].append((epoch, step, layer_norm))
            
    def _analyze_weights(self, model: nn.Module, epoch: int, step: int):
        """Analyze weight distributions and detect issues."""
        for name, param in model.named_parameters():
            if param.data is not None:
                weight_data = param.data.cpu().numpy()
                
                # Basic statistics
                mean_val = np.mean(weight_data)
                std_val = np.std(weight_data)
                max_val = np.max(np.abs(weight_data))
                
                self.weight_history[f'{name}_mean'].append((epoch, step, mean_val))
                self.weight_history[f'{name}_std'].append((epoch, step, std_val))
                self.weight_history[f'{name}_max_abs'].append((epoch, step, max_val))
                
                # Dead neuron detection (for linear layers)
                if 'weight' in name and len(weight_data.shape) == 2:
                    # Check for neurons with very small weights
                    neuron_norms = np.linalg.norm(weight_data, axis=1)
                    dead_threshold = 1e-6
                    dead_count = np.sum(neuron_norms < dead_threshold)
                    
                    if dead_count > 0:
                        if name not in self.dead_neurons:
                            self.dead_neurons[name] = []
                        self.dead_neurons[name].append({
                            'epoch': epoch,
                            'step': step,
                            'dead_count': dead_count,
                            'total_neurons': weight_data.shape[0],
                            'dead_ratio': dead_count / weight_data.shape[0]
                        })
                        
    def _setup_activation_hooks(self, model: nn.Module):
        """Setup hooks to track activation statistics."""
        if not self.track_activations:
            return
            
        # Clear existing hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks.clear()
        
        def activation_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    output_data = output.detach().cpu().numpy()
                    
                    # Basic statistics
                    mean_activation = np.mean(output_data)
                    std_activation = np.std(output_data)
                    max_activation = np.max(np.abs(output_data))
                    
                    # Sparsity (fraction of near-zero activations)
                    sparsity = np.mean(np.abs(output_data) < 1e-6)
                    
                    timestamp = time.time()
                    self.activation_history[f'{name}_mean'].append((timestamp, mean_activation))
                    self.activation_history[f'{name}_std'].append((timestamp, std_activation))
                    self.activation_history[f'{name}_max_abs'].append((timestamp, max_activation))
                    self.activation_history[f'{name}_sparsity'].append((timestamp, sparsity))
                    
            return hook_fn
            
        # Register hooks for key layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)):
                hook = module.register_forward_hook(activation_hook(name))
                self.activation_hooks.append(hook)
                
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive model health report."""
        report = {
            'gradient_health': self._analyze_gradient_health(),
            'weight_health': self._analyze_weight_health(),
            'dead_neurons': dict(self.dead_neurons),
            'gradient_issues': dict(self.gradient_issues),
            'recommendations': []
        }
        
        # Add activation health if tracking
        if self.track_activations and self.activation_history:
            report['activation_health'] = self._analyze_activation_health()
            
        # Generate recommendations
        report['recommendations'] = self._generate_health_recommendations(report)
        
        return report
        
    def _analyze_gradient_health(self) -> Dict[str, Any]:
        """Analyze gradient health."""
        if 'total_norm' not in self.gradient_history:
            return {}
            
        total_norms = [norm for _, _, norm in self.gradient_history['total_norm']]
        
        return {
            'mean_gradient_norm': np.mean(total_norms),
            'std_gradient_norm': np.std(total_norms),
            'max_gradient_norm': np.max(total_norms),
            'min_gradient_norm': np.min(total_norms),
            'gradient_explosion_risk': np.max(total_norms) > 10.0,
            'gradient_vanishing_risk': np.min(total_norms) < 1e-6,
            'gradient_stability': np.std(total_norms) / (np.mean(total_norms) + 1e-8)
        }
        
    def _analyze_weight_health(self) -> Dict[str, Any]:
        """Analyze weight health."""
        weight_stats = {}
        
        for key, history in self.weight_history.items():
            if '_std' in key:  # Focus on weight standard deviations
                layer_name = key.replace('_std', '')
                values = [val for _, _, val in history]
                
                weight_stats[layer_name] = {
                    'mean_std': np.mean(values),
                    'std_trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'stable',
                    'weight_decay_effective': np.mean(values) < 1.0  # Arbitrary threshold
                }
                
        return weight_stats
        
    def _analyze_activation_health(self) -> Dict[str, Any]:
        """Analyze activation health."""
        activation_stats = {}
        
        for key, history in self.activation_history.items():
            if '_sparsity' in key:
                layer_name = key.replace('_sparsity', '')
                sparsity_values = [val for _, val in history]
                
                activation_stats[layer_name] = {
                    'mean_sparsity': np.mean(sparsity_values),
                    'dead_neurons_risk': np.mean(sparsity_values) > 0.9,  # High sparsity
                    'activation_diversity': 1.0 - np.mean(sparsity_values)
                }
                
        return activation_stats
        
    def _generate_health_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health analysis."""
        recommendations = []
        
        # Gradient-based recommendations
        gradient_health = report.get('gradient_health', {})
        if gradient_health.get('gradient_explosion_risk', False):
            recommendations.append("Consider gradient clipping to prevent gradient explosion")
        if gradient_health.get('gradient_vanishing_risk', False):
            recommendations.append("Consider using residual connections or different activation functions")
        if gradient_health.get('gradient_stability', 0) > 1.0:
            recommendations.append("Gradient norms are unstable, consider adjusting learning rate")
            
        # Dead neuron recommendations
        if self.dead_neurons:
            total_dead = sum(len(issues) for issues in self.dead_neurons.values())
            if total_dead > 0:
                recommendations.append(f"Detected {total_dead} instances of dead neurons, consider different initialization or activation functions")
                
        # Activation-based recommendations
        activation_health = report.get('activation_health', {})
        for layer, stats in activation_health.items():
            if stats.get('dead_neurons_risk', False):
                recommendations.append(f"Layer {layer} has high sparsity, risk of dead neurons")
                
        return recommendations
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of model health monitoring."""
        return self.get_health_report()
        
    def reset(self):
        """Reset model health monitoring."""
        self.gradient_history.clear()
        self.weight_history.clear()
        if self.activation_history:
            self.activation_history.clear()
        self.dead_neurons.clear()
        self.gradient_issues.clear()
        
        # Remove activation hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks.clear()


class TrainingMonitor:
    """
    Comprehensive training monitor that orchestrates all monitoring components.
    
    Provides unified interface for tracking metrics, resources, and model health
    with real-time alerting and dashboard capabilities.
    """
    
    def __init__(
        self,
        save_dir: str,
        enable_resource_monitoring: bool = True,
        enable_model_health: bool = True,
        enable_alerts: bool = True,
        track_activations: bool = False,
        dashboard_port: Optional[int] = None
    ):
        """
        Initialize comprehensive training monitor.
        
        Args:
            save_dir: Directory to save monitoring data
            enable_resource_monitoring: Whether to enable resource monitoring
            enable_model_health: Whether to enable model health monitoring
            enable_alerts: Whether to enable alerting system
            track_activations: Whether to track activation statistics
            dashboard_port: Port for real-time dashboard (if available)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize monitoring components
        self.metric_tracker = MetricTracker()
        
        self.resource_monitor = None
        if enable_resource_monitoring:
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.start_monitoring()
            
        self.model_health_monitor = None
        if enable_model_health:
            self.model_health_monitor = ModelHealthMonitor(track_activations=track_activations)
            
        # Alerting system
        self.alerts_enabled = enable_alerts
        self.alert_configs = []
        self.alert_history = []
        
        # Dashboard
        self.dashboard_port = dashboard_port
        self.dashboard_thread = None
        
        # Training state
        self.training_start_time = None
        self.current_epoch = 0
        self.current_step = 0
        
    def start_training(self):
        """Called at the start of training."""
        self.training_start_time = time.time()
        logger.info("Training monitoring started")
        
        # Start dashboard if requested
        if self.dashboard_port:
            self._start_dashboard()
            
    def end_training(self):
        """Called at the end of training."""
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
            
        # Stop dashboard
        if self.dashboard_thread:
            # Dashboard stopping would be implemented here
            pass
            
        # Save final monitoring data
        self._save_monitoring_data()
        
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        logger.info(f"Training monitoring ended. Total time: {training_time:.2f} seconds")
        
    def update(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None
    ):
        """
        Update all monitoring components.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Training/validation metrics
            model: Model for health monitoring (optional)
        """
        self.current_epoch = epoch
        self.current_step = step
        
        # Update metric tracker
        system_metrics = None
        if self.resource_monitor:
            system_metrics = self.resource_monitor.get_current_resources()
            
        model_metrics = None
        if self.model_health_monitor and model:
            self.model_health_monitor.update(model, epoch, step)
            model_metrics = self._extract_model_metrics()
            
        self.metric_tracker.update(
            epoch=epoch,
            step=step,
            metrics=metrics,
            system_metrics=system_metrics,
            model_metrics=model_metrics
        )
        
        # Check alerts
        if self.alerts_enabled:
            self._check_alerts(metrics, system_metrics, model_metrics)
            
    def _extract_model_metrics(self) -> Dict[str, float]:
        """Extract key model health metrics."""
        if not self.model_health_monitor:
            return {}
            
        health_report = self.model_health_monitor.get_health_report()
        
        model_metrics = {}
        
        # Gradient health
        gradient_health = health_report.get('gradient_health', {})
        if 'mean_gradient_norm' in gradient_health:
            model_metrics['gradient_norm'] = gradient_health['mean_gradient_norm']
        if 'gradient_stability' in gradient_health:
            model_metrics['gradient_stability'] = gradient_health['gradient_stability']
            
        # Dead neurons
        dead_neurons = health_report.get('dead_neurons', {})
        total_dead = sum(len(issues) for issues in dead_neurons.values())
        model_metrics['dead_neurons_count'] = total_dead
        
        return model_metrics
        
    def add_alert(
        self,
        metric_name: str,
        threshold: float,
        condition: str = "greater",
        consecutive_violations: int = 3
    ):
        """Add an alert configuration."""
        alert_config = AlertConfig(
            metric_name=metric_name,
            threshold=threshold,
            condition=condition,
            consecutive_violations=consecutive_violations
        )
        self.alert_configs.append(alert_config)
        
    def _check_alerts(
        self,
        metrics: Dict[str, float],
        system_metrics: Optional[Dict[str, float]],
        model_metrics: Optional[Dict[str, float]]
    ):
        """Check all alert conditions."""
        all_metrics = {**metrics}
        if system_metrics:
            all_metrics.update(system_metrics)
        if model_metrics:
            all_metrics.update(model_metrics)
            
        for alert_config in self.alert_configs:
            if not alert_config.enabled:
                continue
                
            metric_name = alert_config.metric_name
            if metric_name not in all_metrics:
                continue
                
            value = all_metrics[metric_name]
            threshold = alert_config.threshold
            condition = alert_config.condition
            
            # Check condition
            violation = False
            if condition == "greater" and value > threshold:
                violation = True
            elif condition == "less" and value < threshold:
                violation = True
            elif condition == "equal" and abs(value - threshold) < 1e-6:
                violation = True
                
            if violation:
                alert = {
                    'timestamp': time.time(),
                    'epoch': self.current_epoch,
                    'step': self.current_step,
                    'metric_name': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'condition': condition,
                    'message': f"Alert: {metric_name} = {value:.4f} ({condition} {threshold})"
                }
                
                self.alert_history.append(alert)
                logger.warning(alert['message'])
                
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary from all monitoring components."""
        summary = {
            'training_info': {
                'current_epoch': self.current_epoch,
                'current_step': self.current_step,
                'training_time': time.time() - self.training_start_time if self.training_start_time else 0
            },
            'metrics': self.metric_tracker.get_summary(),
            'alerts': {
                'total_alerts': len(self.alert_history),
                'recent_alerts': self.alert_history[-10:] if self.alert_history else [],
                'alert_configs': [asdict(config) for config in self.alert_configs]
            }
        }
        
        if self.resource_monitor:
            summary['resources'] = self.resource_monitor.get_summary()
            
        if self.model_health_monitor:
            summary['model_health'] = self.model_health_monitor.get_summary()
            
        return summary
        
    def _save_monitoring_data(self):
        """Save all monitoring data to disk."""
        summary = self.get_comprehensive_summary()
        
        # Save summary
        summary_path = self.save_dir / 'monitoring_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Save detailed data
        detailed_data = {
            'metric_snapshots': [asdict(snapshot) for snapshot in self.metric_tracker.snapshots],
            'alert_history': self.alert_history
        }
        
        if self.resource_monitor:
            detailed_data['resource_history'] = dict(self.resource_monitor.resource_history)
            
        if self.model_health_monitor:
            detailed_data['gradient_history'] = dict(self.model_health_monitor.gradient_history)
            detailed_data['weight_history'] = dict(self.model_health_monitor.weight_history)
            
        detailed_path = self.save_dir / 'monitoring_detailed.json'
        with open(detailed_path, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
            
        logger.info(f"Monitoring data saved to {self.save_dir}")
        
    def _start_dashboard(self):
        """Start real-time monitoring dashboard."""
        # This would start a web server for real-time monitoring
        # Implementation would depend on chosen web framework (Flask, FastAPI, etc.)
        logger.info(f"Dashboard would start on port {self.dashboard_port}")
        
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Training metrics
            ax1 = axes[0, 0]
            for metric_name in ['train_loss', 'val_loss']:
                if metric_name in self.metric_tracker.metrics_history:
                    values = self.metric_tracker.metrics_history[metric_name]
                    ax1.plot(values, label=metric_name)
            ax1.set_title('Training Progress')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Resource usage
            ax2 = axes[0, 1]
            if self.resource_monitor:
                for resource_name in ['cpu_percent', 'memory_percent']:
                    if resource_name in self.resource_monitor.resource_history:
                        history = self.resource_monitor.resource_history[resource_name]
                        timestamps, values = zip(*history)
                        ax2.plot(values, label=resource_name)
                ax2.set_title('Resource Usage')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Percentage')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
            # Plot 3: Model health
            ax3 = axes[1, 0]
            if self.model_health_monitor and 'total_norm' in self.model_health_monitor.gradient_history:
                history = self.model_health_monitor.gradient_history['total_norm']
                epochs, steps, norms = zip(*history)
                ax3.plot(norms, label='Gradient Norm')
                ax3.set_title('Model Health')
                ax3.set_xlabel('Updates')
                ax3.set_ylabel('Gradient Norm')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
            # Plot 4: Alerts over time
            ax4 = axes[1, 1]
            if self.alert_history:
                alert_times = [alert['timestamp'] for alert in self.alert_history]
                alert_counts = list(range(1, len(alert_times) + 1))
                ax4.plot(alert_counts, label='Cumulative Alerts')
                ax4.set_title('Alert History')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Cumulative Alert Count')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            
    def reset_all(self):
        """Reset all monitoring components."""
        self.metric_tracker.reset()
        if self.resource_monitor:
            self.resource_monitor.reset()
        if self.model_health_monitor:
            self.model_health_monitor.reset()
        self.alert_history.clear()
        self.training_start_time = None


def create_monitoring_dashboard(monitor: TrainingMonitor, port: int = 8080):
    """
    Create real-time monitoring dashboard.
    
    Args:
        monitor: TrainingMonitor instance
        port: Port for the dashboard server
        
    Returns:
        Dashboard server instance (implementation dependent)
    """
    # This would create a web-based dashboard for real-time monitoring
    # Implementation would use Flask, FastAPI, or similar web framework
    logger.info(f"Dashboard creation requested on port {port}")
    return None  # Placeholder


def generate_training_report(monitor: TrainingMonitor, save_path: str):
    """
    Generate comprehensive training report.
    
    Args:
        monitor: TrainingMonitor instance
        save_path: Path to save the report
    """
    summary = monitor.get_comprehensive_summary()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Monitoring Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 30px; }}
            .metric {{ margin: 10px 0; }}
            .alert {{ background-color: #ffebee; padding: 10px; margin: 5px 0; border-radius: 4px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Training Monitoring Report</h1>
        
        <div class="section">
            <h2>Training Overview</h2>
            <div class="metric">Current Epoch: {summary['training_info']['current_epoch']}</div>
            <div class="metric">Current Step: {summary['training_info']['current_step']}</div>
            <div class="metric">Training Time: {summary['training_info']['training_time']:.2f} seconds</div>
        </div>
        
        <div class="section">
            <h2>Latest Metrics</h2>
            {_format_metrics_html(summary['metrics'].get('latest_metrics', {}))}
        </div>
        
        <div class="section">
            <h2>Alerts</h2>
            <div class="metric">Total Alerts: {summary['alerts']['total_alerts']}</div>
            {_format_alerts_html(summary['alerts'].get('recent_alerts', []))}
        </div>
        
        <div class="section">
            <h2>Resource Usage</h2>
            {_format_resources_html(summary.get('resources', {}))}
        </div>
        
        <div class="section">
            <h2>Model Health</h2>
            {_format_model_health_html(summary.get('model_health', {}))}
        </div>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)
        
    logger.info(f"Training report saved to {save_path}")


def _format_metrics_html(metrics: Dict[str, float]) -> str:
    """Format metrics for HTML report."""
    if not metrics:
        return "<p>No metrics available</p>"
        
    html = "<table><tr><th>Metric</th><th>Value</th></tr>"
    for name, value in metrics.items():
        html += f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
    html += "</table>"
    return html


def _format_alerts_html(alerts: List[Dict[str, Any]]) -> str:
    """Format alerts for HTML report."""
    if not alerts:
        return "<p>No recent alerts</p>"
        
    html = ""
    for alert in alerts:
        html += f'<div class="alert">{alert["message"]} (Epoch {alert["epoch"]}, Step {alert["step"]})</div>'
    return html


def _format_resources_html(resources: Dict[str, Any]) -> str:
    """Format resource usage for HTML report."""
    if not resources:
        return "<p>Resource monitoring not available</p>"
        
    current = resources.get('current_resources', {})
    if not current:
        return "<p>No current resource data</p>"
        
    html = "<table><tr><th>Resource</th><th>Current Value</th></tr>"
    for name, value in current.items():
        if isinstance(value, float):
            html += f"<tr><td>{name}</td><td>{value:.2f}</td></tr>"
        else:
            html += f"<tr><td>{name}</td><td>{value}</td></tr>"
    html += "</table>"
    return html


def _format_model_health_html(model_health: Dict[str, Any]) -> str:
    """Format model health for HTML report."""
    if not model_health:
        return "<p>Model health monitoring not available</p>"
        
    gradient_health = model_health.get('gradient_health', {})
    if not gradient_health:
        return "<p>No model health data</p>"
        
    html = "<table><tr><th>Health Indicator</th><th>Value</th></tr>"
    for name, value in gradient_health.items():
        if isinstance(value, (int, float)):
            html += f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
        else:
            html += f"<tr><td>{name}</td><td>{value}</td></tr>"
    html += "</table>"
    return html


def detect_training_issues(monitor: TrainingMonitor) -> List[Dict[str, Any]]:
    """
    Detect potential training issues from monitoring data.
    
    Args:
        monitor: TrainingMonitor instance
        
    Returns:
        List of detected issues
    """
    issues = []
    summary = monitor.get_comprehensive_summary()
    
    # Check for training stagnation
    metrics = summary.get('metrics', {})
    trends = metrics.get('trends', {})
    
    for metric_name, trend_data in trends.items():
        if 'train_loss' in metric_name or 'val_loss' in metric_name:
            if trend_data.get('slope', 0) > -1e-6:  # Loss not decreasing
                issues.append({
                    'type': 'training_stagnation',
                    'metric': metric_name,
                    'description': f'{metric_name} is not decreasing',
                    'severity': 'medium'
                })
                
    # Check for overfitting
    if 'train_loss' in trends and 'val_loss' in trends:
        train_trend = trends['train_loss'].get('slope', 0)
        val_trend = trends['val_loss'].get('slope', 0)
        
        if train_trend < -1e-4 and val_trend > 1e-4:  # Train loss decreasing, val loss increasing
            issues.append({
                'type': 'overfitting',
                'description': 'Training loss decreasing while validation loss increasing',
                'severity': 'high'
            })
            
    # Check resource issues
    resources = summary.get('resources', {})
    current_resources = resources.get('current_resources', {})
    
    if current_resources.get('memory_percent', 0) > 90:
        issues.append({
            'type': 'high_memory_usage',
            'value': current_resources['memory_percent'],
            'description': 'Memory usage is very high',
            'severity': 'high'
        })
        
    # Check model health issues
    model_health = summary.get('model_health', {})
    gradient_health = model_health.get('gradient_health', {})
    
    if gradient_health.get('gradient_explosion_risk', False):
        issues.append({
            'type': 'gradient_explosion',
            'description': 'Risk of gradient explosion detected',
            'severity': 'high'
        })
        
    if gradient_health.get('gradient_vanishing_risk', False):
        issues.append({
            'type': 'gradient_vanishing',
            'description': 'Risk of vanishing gradients detected',
            'severity': 'medium'
        })
        
    return issues
