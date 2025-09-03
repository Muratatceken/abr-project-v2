"""
Ensemble methods framework for ABR transformer models.

This module implements various ensemble strategies including:
- Model ensemble with weighted averaging
- Snapshot ensemble from single training run
- Cross-validation ensemble from multiple folds
- Uncertainty quantification through ensemble variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    """
    Abstract base class for ensemble methods.
    """
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make ensemble predictions."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty estimates."""
        pass
    
    @abstractmethod
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """Add a model to the ensemble."""
        pass


class ModelEnsemble(BaseEnsemble):
    """
    Standard model ensemble with weighted averaging of predictions.
    
    Supports both signal generation and classification tasks with
    uncertainty quantification through ensemble variance.
    """
    
    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        weights: Optional[List[float]] = None,
        device: str = 'cpu',
        temperature: float = 1.0
    ):
        """
        Initialize model ensemble.
        
        Args:
            models: List of models to ensemble
            weights: Weights for each model (normalized automatically)
            device: Device to run ensemble on
            temperature: Temperature for calibration
        """
        self.models = models or []
        self.weights = weights or []
        self.device = device
        self.temperature = temperature
        
        # Normalize weights
        if self.weights and len(self.weights) == len(self.models):
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        else:
            self.weights = [1.0 / len(self.models) for _ in self.models] if self.models else []
            
        # Move models to device and set to eval mode
        for model in self.models:
            model.to(device)
            model.eval()
            
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """Add a model to the ensemble."""
        model.to(self.device)
        model.eval()
        self.models.append(model)
        self.weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make ensemble predictions by averaging model outputs.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
            
        x = x.to(self.device)
        ensemble_outputs = {}
        
        with torch.no_grad():
            # Collect predictions from all models
            all_predictions = []
            for model in self.models:
                pred = model(x)
                all_predictions.append(pred)
                
            # Initialize ensemble outputs with first prediction structure
            if isinstance(all_predictions[0], dict):
                # Multi-output model (e.g., signal + classification)
                for key in all_predictions[0].keys():
                    weighted_sum = torch.zeros_like(all_predictions[0][key])
                    for i, pred in enumerate(all_predictions):
                        if key in pred:
                            weighted_sum += self.weights[i] * pred[key]
                    ensemble_outputs[key] = weighted_sum / self.temperature
            else:
                # Single output model
                weighted_sum = torch.zeros_like(all_predictions[0])
                for i, pred in enumerate(all_predictions):
                    weighted_sum += self.weights[i] * pred
                ensemble_outputs['prediction'] = weighted_sum / self.temperature
                
        return ensemble_outputs
        
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates using ensemble variance.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        if not self.models:
            raise ValueError("No models in ensemble")
            
        x = x.to(self.device)
        
        with torch.no_grad():
            # Collect predictions from all models
            all_predictions = []
            for model in self.models:
                pred = model(x)
                all_predictions.append(pred)
                
            results = {}
            
            if isinstance(all_predictions[0], dict):
                # Multi-output model
                for key in all_predictions[0].keys():
                    predictions = torch.stack([pred[key] for pred in all_predictions if key in pred])
                    
                    # Compute mean and variance
                    mean_pred = torch.mean(predictions, dim=0)
                    var_pred = torch.var(predictions, dim=0)
                    std_pred = torch.sqrt(var_pred + 1e-8)
                    
                    results[f'{key}_mean'] = mean_pred / self.temperature
                    results[f'{key}_std'] = std_pred
                    results[f'{key}_variance'] = var_pred
                    
                    # Compute confidence intervals (assuming normal distribution)
                    results[f'{key}_ci_lower'] = mean_pred - 1.96 * std_pred
                    results[f'{key}_ci_upper'] = mean_pred + 1.96 * std_pred
            else:
                # Single output model
                predictions = torch.stack(all_predictions)
                
                mean_pred = torch.mean(predictions, dim=0)
                var_pred = torch.var(predictions, dim=0)
                std_pred = torch.sqrt(var_pred + 1e-8)
                
                results['prediction_mean'] = mean_pred / self.temperature
                results['prediction_std'] = std_pred
                results['prediction_variance'] = var_pred
                results['prediction_ci_lower'] = mean_pred - 1.96 * std_pred
                results['prediction_ci_upper'] = mean_pred + 1.96 * std_pred
                
        return results
        
    def calibrate_temperature(self, val_loader, criterion=None):
        """
        Calibrate ensemble temperature using validation data.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function for calibration
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        # Find optimal temperature using grid search
        temperatures = np.linspace(0.1, 5.0, 50)
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in temperatures:
            self.temperature = temp
            total_loss = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x, y = _extract_xy(batch)
                        
                    predictions = self.predict(x)
                    
                    # Use first prediction for calibration
                    pred_key = list(predictions.keys())[0]
                    loss = criterion(predictions[pred_key], y)
                    
                    total_loss += loss.item() * x.size(0)
                    total_samples += x.size(0)
                    
            avg_loss = total_loss / total_samples
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_temp = temp
                
        self.temperature = best_temp
        logger.info(f"Optimal temperature found: {best_temp:.3f}")


class SnapshotEnsemble(BaseEnsemble):
    """
    Snapshot ensemble that collects model snapshots during training.
    
    Uses cosine annealing learning rate schedule to encourage diverse
    snapshots at different points in the loss landscape.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        snapshot_epochs: List[int],
        device: str = 'cpu',
        max_snapshots: int = 10
    ):
        """
        Initialize snapshot ensemble.
        
        Args:
            base_model: Base model architecture
            snapshot_epochs: Epochs at which to take snapshots
            device: Device to run ensemble on
            max_snapshots: Maximum number of snapshots to keep
        """
        self.base_model = base_model
        self.snapshot_epochs = sorted(snapshot_epochs)
        self.device = device
        self.max_snapshots = max_snapshots
        
        self.snapshots = []
        self.snapshot_metadata = []
        
    def should_take_snapshot(self, epoch: int) -> bool:
        """Check if a snapshot should be taken at this epoch."""
        return epoch in self.snapshot_epochs
        
    def take_snapshot(
        self, 
        model: nn.Module, 
        epoch: int, 
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Take a snapshot of the current model.
        
        Args:
            model: Current model to snapshot
            epoch: Current epoch
            metrics: Optional metrics for this snapshot
        """
        # Create a deep copy of the model
        snapshot = copy.deepcopy(model)
        snapshot.to(self.device)
        snapshot.eval()
        
        # Store snapshot with metadata
        self.snapshots.append(snapshot)
        self.snapshot_metadata.append({
            'epoch': epoch,
            'metrics': metrics or {},
            'timestamp': torch.cuda.current_stream().query() if torch.cuda.is_available() else 0
        })
        
        # Remove oldest snapshots if we exceed max_snapshots
        if len(self.snapshots) > self.max_snapshots:
            # Keep the best snapshots based on validation metric
            if all('val_loss' in meta['metrics'] for meta in self.snapshot_metadata):
                # Sort by validation loss and keep best ones
                sorted_indices = sorted(
                    range(len(self.snapshots)), 
                    key=lambda i: self.snapshot_metadata[i]['metrics'].get('val_loss', float('inf'))
                )
                
                # Keep best snapshots
                keep_indices = sorted_indices[:self.max_snapshots]
                self.snapshots = [self.snapshots[i] for i in keep_indices]
                self.snapshot_metadata = [self.snapshot_metadata[i] for i in keep_indices]
            else:
                # Remove oldest
                self.snapshots.pop(0)
                self.snapshot_metadata.pop(0)
                
        logger.info(f"Took snapshot at epoch {epoch}. Total snapshots: {len(self.snapshots)}")
        
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """Add a model snapshot manually."""
        model.to(self.device)
        model.eval()
        self.snapshots.append(model)
        self.snapshot_metadata.append({
            'epoch': -1,
            'metrics': {},
            'weight': weight
        })
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make ensemble predictions using all snapshots."""
        if not self.snapshots:
            raise ValueError("No snapshots available")
            
        x = x.to(self.device)
        
        with torch.no_grad():
            all_predictions = []
            for snapshot in self.snapshots:
                pred = snapshot(x)
                all_predictions.append(pred)
                
            # Average predictions
            ensemble_outputs = {}
            if isinstance(all_predictions[0], dict):
                for key in all_predictions[0].keys():
                    avg_pred = torch.stack([pred[key] for pred in all_predictions if key in pred]).mean(0)
                    ensemble_outputs[key] = avg_pred
            else:
                ensemble_outputs['prediction'] = torch.stack(all_predictions).mean(0)
                
        return ensemble_outputs
        
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty estimates."""
        if not self.snapshots:
            raise ValueError("No snapshots available")
            
        x = x.to(self.device)
        
        with torch.no_grad():
            all_predictions = []
            for snapshot in self.snapshots:
                pred = snapshot(x)
                all_predictions.append(pred)
                
            results = {}
            if isinstance(all_predictions[0], dict):
                for key in all_predictions[0].keys():
                    predictions = torch.stack([pred[key] for pred in all_predictions if key in pred])
                    
                    mean_pred = torch.mean(predictions, dim=0)
                    std_pred = torch.std(predictions, dim=0)
                    
                    results[f'{key}_mean'] = mean_pred
                    results[f'{key}_std'] = std_pred
            else:
                predictions = torch.stack(all_predictions)
                results['prediction_mean'] = torch.mean(predictions, dim=0)
                results['prediction_std'] = torch.std(predictions, dim=0)
                
        return results
        
    def get_diversity_score(self, data_loader=None, seq_len: int = 200) -> float:
        """
        Compute diversity score of snapshots using prediction disagreement.
        
        Args:
            data_loader: Optional data loader to use real data for diversity computation
            seq_len: Sequence length for dummy input if data_loader is None (default 200 for ABR)
        
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(self.snapshots) < 2:
            return 0.0
            
        if data_loader is not None:
            # Use real data from data loader
            batch = next(iter(data_loader))
            if isinstance(batch, dict):
                # Handle ABR dataset format
                dummy_input = batch.get('x0', batch.get('signal', torch.randn(10, 1, seq_len)))
            else:
                dummy_input = batch[0] if isinstance(batch, (list, tuple)) else batch
            dummy_input = dummy_input.to(self.device)
        else:
            # Generate dummy inputs with correct ABR shape
            dummy_input = torch.randn(10, 1, seq_len).to(self.device)
        
        with torch.no_grad():
            predictions = []
            for snapshot in self.snapshots:
                pred = snapshot(dummy_input)
                if isinstance(pred, dict):
                    # Use first output for diversity computation
                    pred = pred[list(pred.keys())[0]]
                predictions.append(pred.flatten())
                
            # Compute pairwise disagreement
            predictions = torch.stack(predictions)
            pairwise_diffs = []
            
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    diff = torch.mean((predictions[i] - predictions[j]) ** 2)
                    pairwise_diffs.append(diff.item())
                    
            return np.mean(pairwise_diffs) if pairwise_diffs else 0.0


class CrossValidationEnsemble(BaseEnsemble):
    """
    Ensemble from cross-validation folds with patient-stratified splitting.
    
    Combines models trained on different CV folds to create a robust
    ensemble with confidence intervals.
    """
    
    def __init__(
        self,
        fold_models: Optional[List[nn.Module]] = None,
        fold_metrics: Optional[List[Dict[str, float]]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize cross-validation ensemble.
        
        Args:
            fold_models: Models trained on different CV folds
            fold_metrics: Metrics for each fold model
            device: Device to run ensemble on
        """
        self.fold_models = fold_models or []
        self.fold_metrics = fold_metrics or []
        self.device = device
        
        # Move models to device and set to eval mode
        for model in self.fold_models:
            model.to(device)
            model.eval()
            
        # Compute fold weights based on performance
        self.fold_weights = self._compute_fold_weights()
        
    def add_fold_model(
        self, 
        model: nn.Module, 
        fold_metrics: Dict[str, float],
        fold_id: int
    ):
        """Add a model from a specific CV fold."""
        model.to(self.device)
        model.eval()
        self.fold_models.append(model)
        self.fold_metrics.append({**fold_metrics, 'fold_id': fold_id})
        
        # Recompute weights
        self.fold_weights = self._compute_fold_weights()
        
    def _compute_fold_weights(self) -> List[float]:
        """Compute weights for fold models based on performance."""
        if not self.fold_metrics:
            return [1.0 / len(self.fold_models) for _ in self.fold_models]
            
        # Use validation loss or accuracy for weighting
        if 'val_loss' in self.fold_metrics[0]:
            # Lower loss = higher weight
            losses = [metrics['val_loss'] for metrics in self.fold_metrics]
            # Convert to weights (inverse of loss)
            weights = [1.0 / (loss + 1e-8) for loss in losses]
        elif 'val_acc' in self.fold_metrics[0]:
            # Higher accuracy = higher weight
            weights = [metrics['val_acc'] for metrics in self.fold_metrics]
        else:
            # Equal weights
            weights = [1.0 for _ in self.fold_models]
            
        # Normalize weights
        total_weight = sum(weights)
        return [w / total_weight for w in weights] if total_weight > 0 else weights
        
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """Add a model to the ensemble (compatibility method)."""
        self.add_fold_model(model, {'manual_weight': weight}, len(self.fold_models))
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make ensemble predictions using weighted averaging."""
        if not self.fold_models:
            raise ValueError("No fold models available")
            
        x = x.to(self.device)
        
        with torch.no_grad():
            ensemble_outputs = {}
            
            for i, model in enumerate(self.fold_models):
                pred = model(x)
                weight = self.fold_weights[i]
                
                if isinstance(pred, dict):
                    for key, value in pred.items():
                        if key not in ensemble_outputs:
                            ensemble_outputs[key] = torch.zeros_like(value)
                        ensemble_outputs[key] += weight * value
                else:
                    if 'prediction' not in ensemble_outputs:
                        ensemble_outputs['prediction'] = torch.zeros_like(pred)
                    ensemble_outputs['prediction'] += weight * pred
                    
        return ensemble_outputs
        
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty estimates and confidence intervals."""
        if not self.fold_models:
            raise ValueError("No fold models available")
            
        x = x.to(self.device)
        
        with torch.no_grad():
            all_predictions = []
            for model in self.fold_models:
                pred = model(x)
                all_predictions.append(pred)
                
            results = {}
            
            if isinstance(all_predictions[0], dict):
                for key in all_predictions[0].keys():
                    predictions = torch.stack([pred[key] for pred in all_predictions if key in pred])
                    
                    # Compute statistics
                    mean_pred = torch.mean(predictions, dim=0)
                    std_pred = torch.std(predictions, dim=0)
                    
                    # Confidence intervals
                    n_folds = len(predictions)
                    t_value = 1.96 if n_folds > 30 else 2.571  # Approximate t-values
                    margin = t_value * std_pred / np.sqrt(n_folds)
                    
                    results[f'{key}_mean'] = mean_pred
                    results[f'{key}_std'] = std_pred
                    results[f'{key}_ci_lower'] = mean_pred - margin
                    results[f'{key}_ci_upper'] = mean_pred + margin
                    results[f'{key}_n_folds'] = torch.tensor(n_folds)
            else:
                predictions = torch.stack(all_predictions)
                
                mean_pred = torch.mean(predictions, dim=0)
                std_pred = torch.std(predictions, dim=0)
                
                n_folds = len(predictions)
                t_value = 1.96 if n_folds > 30 else 2.571
                margin = t_value * std_pred / np.sqrt(n_folds)
                
                results['prediction_mean'] = mean_pred
                results['prediction_std'] = std_pred
                results['prediction_ci_lower'] = mean_pred - margin
                results['prediction_ci_upper'] = mean_pred + margin
                results['n_folds'] = torch.tensor(n_folds)
                
        return results
        
    def get_fold_performance_summary(self) -> Dict[str, Any]:
        """Get summary of fold performances."""
        if not self.fold_metrics:
            return {}
            
        summary = {}
        
        # Collect all metric names
        all_metrics = set()
        for metrics in self.fold_metrics:
            all_metrics.update(metrics.keys())
            
        for metric in all_metrics:
            values = [metrics.get(metric, np.nan) for metrics in self.fold_metrics]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_folds': len(values)
                }
                
        return summary


class EnsemblePredictor:
    """
    Unified interface for different ensemble types with advanced features.
    
    Supports model calibration, uncertainty quantification, and
    ensemble pruning for optimal performance.
    """
    
    def __init__(
        self,
        ensemble: BaseEnsemble,
        calibration_method: str = 'temperature',
        uncertainty_method: str = 'variance'
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            ensemble: Base ensemble instance
            calibration_method: Method for probability calibration
            uncertainty_method: Method for uncertainty estimation
        """
        self.ensemble = ensemble
        self.calibration_method = calibration_method
        self.uncertainty_method = uncertainty_method
        self.is_calibrated = False
        
    def predict(
        self, 
        x: torch.Tensor, 
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with optional uncertainty quantification.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions and optionally uncertainty
        """
        if return_uncertainty:
            return self.ensemble.predict_with_uncertainty(x)
        else:
            return self.ensemble.predict(x)
            
    def calibrate(self, val_loader, method: str = None):
        """
        Calibrate ensemble predictions using validation data.
        
        Args:
            val_loader: Validation data loader
            method: Calibration method ('temperature', 'platt', 'isotonic')
        """
        method = method or self.calibration_method
        
        if method == 'temperature' and hasattr(self.ensemble, 'calibrate_temperature'):
            self.ensemble.calibrate_temperature(val_loader)
            self.is_calibrated = True
        else:
            logger.warning(f"Calibration method '{method}' not implemented for this ensemble type")
            
    def evaluate_ensemble_quality(self, val_loader) -> Dict[str, float]:
        """
        Evaluate ensemble quality metrics.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = _extract_xy(batch)
                    
                # Get predictions with uncertainty
                pred_dict = self.predict(x, return_uncertainty=True)
                
                # Extract predictions and uncertainties
                for key in pred_dict.keys():
                    if key.endswith('_mean'):
                        all_predictions.append(pred_dict[key])
                        all_targets.append(y)
                        
                        # Get corresponding uncertainty
                        uncertainty_key = key.replace('_mean', '_std')
                        if uncertainty_key in pred_dict:
                            all_uncertainties.append(pred_dict[uncertainty_key])
                            
        if all_predictions:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            # Compute accuracy
            if predictions.shape[-1] > 1:  # Classification
                pred_classes = torch.argmax(predictions, dim=-1)
                target_classes = torch.argmax(targets, dim=-1) if targets.shape[-1] > 1 else targets
                accuracy = (pred_classes == target_classes).float().mean().item()
                metrics['accuracy'] = accuracy
            
            # Compute calibration error if uncertainties available
            if all_uncertainties:
                uncertainties = torch.cat(all_uncertainties, dim=0)
                calibration_error = self._compute_calibration_error(predictions, targets, uncertainties)
                metrics['calibration_error'] = calibration_error
                
                # Compute uncertainty quality (correlation with error)
                errors = torch.abs(predictions - targets).mean(dim=-1)
                uncertainty_scores = uncertainties.mean(dim=-1)
                correlation = torch.corrcoef(torch.stack([errors, uncertainty_scores]))[0, 1].item()
                metrics['uncertainty_correlation'] = correlation
                
        return metrics
        
    def _compute_calibration_error(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        uncertainties: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Compute expected calibration error."""
        # This is a simplified version - full implementation would depend on task type
        uncertainty_scores = uncertainties.mean(dim=-1)
        errors = torch.abs(predictions - targets).mean(dim=-1)
        
        # Bin by uncertainty
        sorted_indices = torch.argsort(uncertainty_scores)
        bin_size = len(sorted_indices) // n_bins
        
        calibration_errors = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_indices)
            bin_indices = sorted_indices[start_idx:end_idx]
            
            if len(bin_indices) > 0:
                bin_uncertainties = uncertainty_scores[bin_indices]
                bin_errors = errors[bin_indices]
                
                avg_uncertainty = bin_uncertainties.mean()
                avg_error = bin_errors.mean()
                
                calibration_errors.append(abs(avg_uncertainty - avg_error))
                
        return np.mean(calibration_errors) if calibration_errors else 0.0


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    model_class: type,
    model_kwargs: Dict[str, Any],
    weights: Optional[List[float]] = None,
    device: str = 'cpu'
) -> ModelEnsemble:
    """
    Create model ensemble from checkpoint files.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        model_class: Class of the model to instantiate
        model_kwargs: Keyword arguments for model instantiation
        weights: Optional weights for each model
        device: Device to load models on
        
    Returns:
        ModelEnsemble instance
    """
    models = []
    
    for checkpoint_path in checkpoint_paths:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model instance
        model = model_class(**model_kwargs)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        models.append(model)
        
    return ModelEnsemble(models=models, weights=weights, device=device)


def evaluate_ensemble_diversity(ensemble: BaseEnsemble, data_loader) -> Dict[str, float]:
    """
    Evaluate diversity of ensemble members.
    
    Args:
        ensemble: Ensemble to evaluate
        data_loader: Data loader for evaluation
        
    Returns:
        Dictionary of diversity metrics
    """
    if not hasattr(ensemble, 'models') and not hasattr(ensemble, 'snapshots') and not hasattr(ensemble, 'fold_models'):
        return {'error': 'Ensemble type not supported for diversity evaluation'}
        
    # Get individual models
    if hasattr(ensemble, 'models'):
        models = ensemble.models
    elif hasattr(ensemble, 'snapshots'):
        models = ensemble.snapshots
    elif hasattr(ensemble, 'fold_models'):
        models = ensemble.fold_models
    else:
        return {'error': 'No models found in ensemble'}
        
    if len(models) < 2:
        return {'diversity_score': 0.0, 'agreement_rate': 1.0}
        
    all_predictions = []
    
    # Collect predictions from all models
    with torch.no_grad():
        for batch in data_loader:
            x, _ = _extract_xy(batch)
                
            batch_predictions = []
            for model in models:
                pred = model(x)
                if isinstance(pred, dict):
                    # Use first output
                    pred = pred[list(pred.keys())[0]]
                batch_predictions.append(pred)
                
            all_predictions.append(torch.stack(batch_predictions))
            
    if not all_predictions:
        return {'error': 'No predictions collected'}
        
    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=1)  # [n_models, n_samples, ...]
    
    # Compute pairwise disagreement
    n_models = predictions.shape[0]
    disagreements = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Compute disagreement between model i and j
            diff = torch.mean((predictions[i] - predictions[j]) ** 2)
            disagreements.append(diff.item())
            
    diversity_score = np.mean(disagreements) if disagreements else 0.0
    
    # Compute agreement rate for classification tasks
    if predictions.shape[-1] > 1:  # Multi-class
        pred_classes = torch.argmax(predictions, dim=-1)
        agreement_counts = []
        
        for sample_idx in range(pred_classes.shape[1]):
            sample_preds = pred_classes[:, sample_idx]
            # Count how many models agree with the mode
            mode_pred = torch.mode(sample_preds)[0]
            agreement = (sample_preds == mode_pred).float().mean()
            agreement_counts.append(agreement.item())
            
        agreement_rate = np.mean(agreement_counts)
    else:
        agreement_rate = 1.0 - diversity_score  # Inverse relationship for regression
        
    return {
        'diversity_score': diversity_score,
        'agreement_rate': agreement_rate,
        'n_models': n_models,
        'pairwise_comparisons': len(disagreements)
    }


def optimize_ensemble_weights(
    ensemble: ModelEnsemble,
    val_loader,
    criterion,
    method: str = 'grid_search'
) -> List[float]:
    """
    Optimize ensemble weights using validation data.
    
    Args:
        ensemble: Model ensemble to optimize
        val_loader: Validation data loader
        criterion: Loss function for optimization
        method: Optimization method ('grid_search', 'gradient_descent')
        
    Returns:
        Optimized weights
    """
    if method == 'grid_search':
        return _optimize_weights_grid_search(ensemble, val_loader, criterion)
    elif method == 'gradient_descent':
        return _optimize_weights_gradient_descent(ensemble, val_loader, criterion)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def _optimize_weights_grid_search(ensemble: ModelEnsemble, val_loader, criterion) -> List[float]:
    """Optimize weights using grid search."""
    n_models = len(ensemble.models)
    if n_models == 1:
        return [1.0]
        
    # Generate weight combinations
    from itertools import product
    
    weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    best_weights = None
    best_loss = float('inf')
    
    # For computational efficiency, limit to reasonable combinations
    if n_models == 2:
        weight_combinations = [(w, 1-w) for w in weight_values if 0 <= 1-w <= 1]
    else:
        # Use a subset of combinations for more than 2 models
        weight_combinations = []
        for _ in range(100):  # Sample random combinations
            weights = np.random.dirichlet(np.ones(n_models))
            weight_combinations.append(tuple(weights))
            
    for weights in weight_combinations:
        if abs(sum(weights) - 1.0) > 1e-6:  # Skip non-normalized weights
            continue
            
        # Set weights and evaluate
        original_weights = ensemble.weights.copy()
        ensemble.weights = list(weights)
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch['signal']
                    y = batch.get('target', batch.get('peaks'))
                else:
                    x, y = batch
                    
                predictions = ensemble.predict(x)
                pred_key = list(predictions.keys())[0]
                loss = criterion(predictions[pred_key], y)
                
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                
        avg_loss = total_loss / total_samples
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = list(weights)
            
        # Restore original weights
        ensemble.weights = original_weights
        
    return best_weights if best_weights else ensemble.weights


def _optimize_weights_gradient_descent(ensemble: ModelEnsemble, val_loader, criterion) -> List[float]:
    """Optimize weights using gradient descent."""
    n_models = len(ensemble.models)
    
    # Initialize learnable weights
    log_weights = torch.zeros(n_models, requires_grad=True)
    optimizer = torch.optim.Adam([log_weights], lr=0.01)
    
    for epoch in range(100):  # Optimization epochs
        total_loss = 0
        
        for batch in val_loader:
            if isinstance(batch, dict):
                x = batch['signal']
                y = batch.get('target', batch.get('peaks'))
            else:
                x, y = batch
                
            optimizer.zero_grad()
            
            # Compute weighted predictions
            weights = F.softmax(log_weights, dim=0)
            weighted_pred = None
            
            for i, model in enumerate(ensemble.models):
                with torch.no_grad():
                    pred = model(x)
                    if isinstance(pred, dict):
                        pred = pred[list(pred.keys())[0]]
                        
                if weighted_pred is None:
                    weighted_pred = weights[i] * pred
                else:
                    weighted_pred += weights[i] * pred
                    
            loss = criterion(weighted_pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 20 == 0:
            logger.info(f"Weight optimization epoch {epoch}, loss: {total_loss:.4f}")
            
    # Return optimized weights
    final_weights = F.softmax(log_weights, dim=0).detach().cpu().numpy()
    return final_weights.tolist()


def _extract_xy(batch):
    """
    Private helper to extract x and y from batch with consistent dataset API.
    
    Args:
        batch: Batch data (dict or tuple)
        
    Returns:
        Tuple of (x, y) where x is input tensor and y is target tensor
    """
    if isinstance(batch, dict):
        # ABR dataset format
        if 'x0' in batch:
            x = batch['x0']
            # Extract y from peak_exists if present, else from meta (class index)
            if 'peak_exists' in batch:
                y = batch['peak_exists']
            elif 'meta' in batch:
                if isinstance(batch['meta'], (list, tuple)):
                    # Extract class index from meta
                    y = torch.tensor([m.get('target', 0) for m in batch['meta']], 
                                   dtype=torch.long, device=x.device)
                else:
                    y = batch['meta']
            else:
                y = None
        else:
            # Legacy format fallback
            x = batch.get('signal', batch.get('x', batch[0] if isinstance(batch, (list, tuple)) else None))
            y = batch.get('target', batch.get('peaks', batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None))
    else:
        # Tuple format
        x, y = batch if len(batch) >= 2 else (batch[0], None)
        
    return x, y


# Unit test for _extract_xy helper
def test_extract_xy():
    """
    Unit test for _extract_xy helper covering both ABR-format and legacy batch dicts.
    """
    # Test ABR format with peak_exists
    abr_batch_peaks = {
        'x0': torch.randn(4, 1, 200),
        'peak_exists': torch.randint(0, 2, (4,)).float(),
        'stat': None
    }
    
    x, y = _extract_xy(abr_batch_peaks)
    assert x.shape == (4, 1, 200), f"Expected x shape (4, 1, 200), got {x.shape}"
    assert y.shape == (4,), f"Expected y shape (4,), got {y.shape}"
    assert y.dtype == torch.float, f"Expected y dtype float, got {y.dtype}"
    
    # Test ABR format with meta (class index)
    abr_batch_meta = {
        'x0': torch.randn(4, 1, 200),
        'meta': [{'target': 0}, {'target': 1}, {'target': 2}, {'target': 0}],
        'stat': None
    }
    
    x, y = _extract_xy(abr_batch_meta)
    assert x.shape == (4, 1, 200), f"Expected x shape (4, 1, 200), got {x.shape}"
    assert y.shape == (4,), f"Expected y shape (4,), got {y.shape}"
    assert y.dtype == torch.long, f"Expected y dtype long, got {y.dtype}"
    assert torch.equal(y, torch.tensor([0, 1, 2, 0])), f"Expected y values [0,1,2,0], got {y}"
    
    # Test legacy format
    legacy_batch = {
        'signal': torch.randn(4, 200),
        'target': torch.randint(0, 2, (4,)).float()
    }
    
    x, y = _extract_xy(legacy_batch)
    assert x.shape == (4, 200), f"Expected x shape (4, 200), got {x.shape}"
    assert y.shape == (4,), f"Expected y shape (4,), got {y.shape}"
    
    # Test tuple format
    tuple_batch = (torch.randn(4, 200), torch.randint(0, 2, (4,)).float())
    
    x, y = _extract_xy(tuple_batch)
    assert x.shape == (4, 200), f"Expected x shape (4, 200), got {x.shape}"
    assert y.shape == (4,), f"Expected y shape (4,), got {y.shape}"
    
    print("_extract_xy helper tests passed!")


if __name__ == "__main__":
    # Run the test when the file is executed directly
    test_extract_xy()
