"""
Hyperparameter optimization framework for ABR transformer models.

This module implements comprehensive HPO strategies including:
- Optuna-based Bayesian optimization with TPE sampling
- Multi-objective optimization for accuracy vs efficiency
- Distributed optimization across multiple workers
- Automated architecture search and hyperparameter tuning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import logging
from pathlib import Path
import json
import pickle
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    
    # Study configuration
    study_name: str = "abr_transformer_hpo"
    storage: Optional[str] = None  # Database URL for distributed optimization
    n_trials: int = 100
    timeout: Optional[int] = None  # Timeout in seconds
    
    # Sampling and pruning
    sampler: str = "tpe"  # "tpe", "random", "cmaes"
    pruner: str = "median"  # "median", "hyperband", "none"
    
    # Optimization objectives
    direction: str = "maximize"  # "maximize", "minimize"
    objectives: List[str] = None  # For multi-objective optimization
    
    # Parallel execution
    n_jobs: int = 1
    distributed: bool = False
    
    # Search space configuration
    search_space_config: Optional[str] = None  # Path to search space YAML
    
    # Trial management
    resume: bool = True
    save_intermediate: bool = True
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["val_combined_score"]


class HyperparameterSpace:
    """
    Define and manage hyperparameter search spaces.
    
    Supports different parameter types with constraints and dependencies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hyperparameter space.
        
        Args:
            config: Configuration dictionary for search space
        """
        self.config = config or {}
        self.space_definitions = {}
        self.conditional_spaces = {}
        
    def add_categorical(
        self,
        name: str,
        choices: List[Any],
        condition: Optional[Callable] = None
    ):
        """Add categorical parameter."""
        self.space_definitions[name] = {
            'type': 'categorical',
            'choices': choices,
            'condition': condition
        }
        
    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None,
        condition: Optional[Callable] = None
    ):
        """Add continuous parameter."""
        self.space_definitions[name] = {
            'type': 'float',
            'low': low,
            'high': high,
            'log': log,
            'step': step,
            'condition': condition
        }
        
    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
        condition: Optional[Callable] = None
    ):
        """Add discrete parameter."""
        self.space_definitions[name] = {
            'type': 'int',
            'low': low,
            'high': high,
            'step': step,
            'log': log,
            'condition': condition
        }
        
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        for name, definition in self.space_definitions.items():
            # Check condition if specified
            if definition.get('condition') is not None:
                if not definition['condition'](params):
                    continue
                    
            param_type = definition['type']
            
            if param_type == 'categorical':
                params[name] = trial.suggest_categorical(name, definition['choices'])
            elif param_type == 'float':
                if definition.get('step') is not None:
                    params[name] = trial.suggest_discrete_uniform(
                        name, definition['low'], definition['high'], definition['step']
                    )
                else:
                    params[name] = trial.suggest_float(
                        name, definition['low'], definition['high'], log=definition.get('log', False)
                    )
            elif param_type == 'int':
                params[name] = trial.suggest_int(
                    name, definition['low'], definition['high'], 
                    step=definition.get('step', 1), log=definition.get('log', False)
                )
                
        return params
        
    def load_from_config(self, config_path: str):
        """Load search space from YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for param_name, param_config in config.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'categorical':
                self.add_categorical(param_name, param_config['choices'])
            elif param_type == 'float':
                self.add_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False),
                    step=param_config.get('step')
                )
            elif param_type == 'int':
                self.add_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1),
                    log=param_config.get('log', False)
                )
                
    def create_default_abr_space(self):
        """Create default search space for ABR transformer."""
        # Model architecture parameters
        self.add_categorical('model_type', ['transformer', 'enhanced_transformer'])
        self.add_int('d_model', 128, 512, step=64)
        self.add_int('n_heads', 4, 16, step=2)
        self.add_int('n_layers', 4, 12, step=2)
        self.add_int('d_ff', 256, 2048, step=256)
        self.add_float('dropout', 0.0, 0.5, step=0.1)
        
        # Training parameters
        self.add_float('learning_rate', 1e-5, 1e-2, log=True)
        self.add_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        self.add_int('batch_size', 8, 64, step=8)
        self.add_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Loss function parameters
        self.add_categorical('use_focal_loss', [True, False])
        self.add_float('focal_alpha', 0.1, 0.9, step=0.1)
        self.add_float('focal_gamma', 0.5, 5.0, step=0.5)
        
        # Augmentation parameters
        self.add_float('noise_std', 0.001, 0.1, log=True)
        self.add_int('time_shift_samples', 0, 10, step=1)
        self.add_float('mixup_prob', 0.0, 0.5, step=0.1)
        self.add_float('cutmix_prob', 0.0, 0.5, step=0.1)
        
        # Regularization parameters
        self.add_float('label_smoothing', 0.0, 0.2, step=0.05)
        self.add_categorical('use_curriculum', [True, False])


class HPOObjective:
    """
    Objective function for hyperparameter optimization.
    
    Handles single and multi-objective optimization with cross-validation
    and early stopping integration.
    """
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        hyperparameter_space: HyperparameterSpace,
        base_config: Dict[str, Any],
        cv_folds: int = 3,
        use_early_stopping: bool = True,
        max_epochs: int = 50,
        device: str = 'cpu'
    ):
        """
        Initialize HPO objective.
        
        Args:
            train_fn: Function to train model with given hyperparameters
            eval_fn: Function to evaluate model
            hyperparameter_space: Hyperparameter search space
            base_config: Base configuration to be updated with suggested params
            cv_folds: Number of cross-validation folds
            use_early_stopping: Whether to use early stopping
            max_epochs: Maximum training epochs
            device: Device for training
        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.hyperparameter_space = hyperparameter_space
        self.base_config = base_config.copy()
        self.cv_folds = cv_folds
        self.use_early_stopping = use_early_stopping
        self.max_epochs = max_epochs
        self.device = device
        
        # Trial tracking
        self.trial_results = {}
        
    def __call__(self, trial: optuna.Trial) -> Union[float, List[float]]:
        """
        Objective function for a single trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value(s) to optimize
        """
        # Suggest hyperparameters
        suggested_params = self.hyperparameter_space.suggest_parameters(trial)
        
        # Update configuration with suggested parameters
        trial_config = self._update_config_with_params(self.base_config, suggested_params)
        
        # Run training and evaluation
        try:
            if self.cv_folds > 1:
                # Cross-validation
                cv_scores = self._run_cv_trial(trial, trial_config)
                objective_value = np.mean(cv_scores)
                
                # Store CV results
                self.trial_results[trial.number] = {
                    'params': suggested_params,
                    'cv_scores': cv_scores,
                    'mean_score': objective_value,
                    'std_score': np.std(cv_scores)
                }
            else:
                # Single training run
                objective_value = self._run_single_trial(trial, trial_config)
                
                self.trial_results[trial.number] = {
                    'params': suggested_params,
                    'score': objective_value
                }
                
            # Report intermediate values for pruning
            trial.report(objective_value, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            return objective_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return worst possible value
            return float('-inf') if trial.study.direction == optuna.study.StudyDirection.MAXIMIZE else float('inf')
            
    def _update_config_with_params(
        self,
        base_config: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update configuration with suggested parameters."""
        config = base_config.copy()
        
        # Map parameters to configuration structure
        param_mapping = {
            # Model parameters
            'd_model': ['model', 'd_model'],
            'n_heads': ['model', 'n_heads'],
            'n_layers': ['model', 'n_layers'],
            'd_ff': ['model', 'd_ff'],
            'dropout': ['model', 'dropout'],
            
            # Training parameters
            'learning_rate': ['trainer', 'learning_rate'],
            'batch_size': ['trainer', 'batch_size'],
            'weight_decay': ['trainer', 'weight_decay'],
            'optimizer': ['trainer', 'optimizer'],
            
            # Loss parameters
            'use_focal_loss': ['focal_loss', 'enabled'],
            'focal_alpha': ['focal_loss', 'alpha'],
            'focal_gamma': ['focal_loss', 'gamma'],
            
            # Augmentation parameters
            'noise_std': ['augmentation', 'noise_std'],
            'time_shift_samples': ['augmentation', 'time_shift_samples'],
            'mixup_prob': ['augmentation', 'mixup_prob'],
            'cutmix_prob': ['augmentation', 'cutmix_prob'],
        }
        
        for param_name, value in params.items():
            if param_name in param_mapping:
                path = param_mapping[param_name]
                self._set_nested_config(config, path, value)
            else:
                # Direct parameter
                config[param_name] = value
                
        return config
        
    def _set_nested_config(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set nested configuration value."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
        
    def _run_cv_trial(self, trial: optuna.Trial, config: Dict[str, Any]) -> List[float]:
        """Run cross-validation trial."""
        cv_scores = []
        
        for fold in range(self.cv_folds):
            # Update config for this fold
            fold_config = config.copy()
            fold_config['fold_idx'] = fold
            fold_config['max_epochs'] = self.max_epochs
            
            # Train model
            model, train_metrics = self.train_fn(fold_config)
            
            # Evaluate model
            eval_metrics = self.eval_fn(model, fold_config)
            
            # Extract objective score
            score = eval_metrics.get('val_combined_score', eval_metrics.get('val_accuracy', 0.0))
            cv_scores.append(score)
            
            # Report intermediate value
            trial.report(score, step=fold)
            
            # Check for pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return cv_scores
        
    def _run_single_trial(self, trial: optuna.Trial, config: Dict[str, Any]) -> float:
        """Run single training trial."""
        config['max_epochs'] = self.max_epochs
        
        # Train model
        model, train_metrics = self.train_fn(config)
        
        # Evaluate model
        eval_metrics = self.eval_fn(model, config)
        
        # Extract objective score
        score = eval_metrics.get('val_combined_score', eval_metrics.get('val_accuracy', 0.0))
        
        return score


class OptunaTuner:
    """
    Optuna-based hyperparameter tuner with advanced features.
    
    Supports distributed optimization, multi-objective optimization,
    and comprehensive result analysis.
    """
    
    def __init__(
        self,
        hpo_config: HPOConfig,
        objective_fn: HPOObjective,
        save_dir: str
    ):
        """
        Initialize Optuna tuner.
        
        Args:
            hpo_config: HPO configuration
            objective_fn: Objective function for optimization
            save_dir: Directory to save results
        """
        self.config = hpo_config
        self.objective_fn = objective_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create study
        self.study = self._create_study()
        
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with specified configuration."""
        # Configure sampler
        if self.config.sampler == "tpe":
            sampler = TPESampler(seed=42)
        elif self.config.sampler == "random":
            sampler = RandomSampler(seed=42)
        elif self.config.sampler == "cmaes":
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = TPESampler(seed=42)
            
        # Configure pruner
        if self.config.pruner == "median":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner == "hyperband":
            pruner = HyperbandPruner(min_resource=1, max_resource=50, reduction_factor=3)
        else:
            pruner = optuna.pruners.NopPruner()
            
        # Create study
        if len(self.config.objectives) == 1:
            # Single-objective optimization
            direction = self.config.direction
        else:
            # Multi-objective optimization
            directions = [self.config.direction] * len(self.config.objectives)
            
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            sampler=sampler,
            pruner=pruner,
            direction=direction if len(self.config.objectives) == 1 else directions,
            load_if_exists=self.config.resume
        )
        
        return study
        
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials")
        
        start_time = time.time()
        
        try:
            if self.config.n_jobs == 1:
                # Sequential optimization
                self.study.optimize(
                    self.objective_fn,
                    n_trials=self.config.n_trials,
                    timeout=self.config.timeout,
                    callbacks=[self._trial_callback] if self.config.save_intermediate else None
                )
            else:
                # Parallel optimization
                self._optimize_parallel()
                
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
            
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Extract results
        results = self._extract_results(optimization_time)
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        return results
        
    def _optimize_parallel(self):
        """Run parallel optimization."""
        if self.config.distributed:
            # Distributed optimization (multiple processes)
            logger.info(f"Running distributed optimization with {self.config.n_jobs} processes")
            
            def run_study():
                study = optuna.load_study(
                    study_name=self.config.study_name,
                    storage=self.config.storage
                )
                study.optimize(
                    self.objective_fn,
                    n_trials=self.config.n_trials // self.config.n_jobs,
                    timeout=self.config.timeout
                )
                
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                futures = [executor.submit(run_study) for _ in range(self.config.n_jobs)]
                for future in futures:
                    future.result()
        else:
            # Thread-based parallelization
            logger.info(f"Running parallel optimization with {self.config.n_jobs} threads")
            
            def run_trials(n_trials):
                self.study.optimize(
                    self.objective_fn,
                    n_trials=n_trials,
                    timeout=self.config.timeout
                )
                
            trials_per_thread = self.config.n_trials // self.config.n_jobs
            with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                futures = [executor.submit(run_trials, trials_per_thread) for _ in range(self.config.n_jobs)]
                for future in futures:
                    future.result()
                    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback function called after each trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} completed with value {trial.value}")
            
            # Save intermediate results
            if self.config.save_intermediate:
                intermediate_results = {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                
                intermediate_path = self.save_dir / f'trial_{trial.number}.json'
                with open(intermediate_path, 'w') as f:
                    json.dump(intermediate_results, f, indent=2)
                    
    def _extract_results(self, optimization_time: float) -> Dict[str, Any]:
        """Extract and format optimization results."""
        results = {
            'study_name': self.config.study_name,
            'n_trials': len(self.study.trials),
            'optimization_time': optimization_time,
            'best_trial': None,
            'best_params': None,
            'best_value': None,
            'trial_history': [],
            'parameter_importance': {},
            'optimization_history': []
        }
        
        if self.study.best_trial is not None:
            results['best_trial'] = self.study.best_trial.number
            results['best_params'] = self.study.best_trial.params
            results['best_value'] = self.study.best_trial.value
            
        # Trial history
        for trial in self.study.trials:
            trial_info = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            results['trial_history'].append(trial_info)
            
        # Optimization history (values over time)
        values = [trial.value for trial in self.study.trials if trial.value is not None]
        results['optimization_history'] = values
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            results['parameter_importance'] = importance
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            
        return results
        
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results."""
        # Save main results
        results_path = self.save_dir / 'hpo_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save best parameters as YAML config
        if results['best_params']:
            best_config_path = self.save_dir / 'best_config.yaml'
            with open(best_config_path, 'w') as f:
                yaml.dump(results['best_params'], f, default_flow_style=False)
                
        # Save study object
        study_path = self.save_dir / 'study.pkl'
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
            
        logger.info(f"Results saved to {self.save_dir}")
        
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best configuration found."""
        if self.study.best_trial is None:
            raise ValueError("No completed trials found")
            
        return self.study.best_trial.params
        
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get Pareto front for multi-objective optimization."""
        if len(self.config.objectives) == 1:
            raise ValueError("Pareto front only available for multi-objective optimization")
            
        pareto_trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pareto_trials.append({
                    'trial_number': trial.number,
                    'values': trial.values,
                    'params': trial.params
                })
                
        return pareto_trials


class AutoML:
    """
    Automated machine learning pipeline for ABR models.
    
    Combines hyperparameter optimization with architecture search
    and automated feature selection.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        search_space_config: Optional[str] = None,
        save_dir: str = "automl_results"
    ):
        """
        Initialize AutoML pipeline.
        
        Args:
            base_config: Base configuration for training
            search_space_config: Path to search space configuration
            save_dir: Directory to save results
        """
        self.base_config = base_config
        self.search_space_config = search_space_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.hyperparameter_space = HyperparameterSpace()
        if search_space_config:
            self.hyperparameter_space.load_from_config(search_space_config)
        else:
            self.hyperparameter_space.create_default_abr_space()
            
    def run_automl(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        n_trials: int = 100,
        cv_folds: int = 3,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run complete AutoML pipeline.
        
        Args:
            train_fn: Training function
            eval_fn: Evaluation function
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with AutoML results
        """
        logger.info("Starting AutoML pipeline")
        
        # Phase 1: Hyperparameter optimization
        logger.info("Phase 1: Hyperparameter optimization")
        hpo_config = HPOConfig(
            study_name="automl_hpo",
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        
        objective_fn = HPOObjective(
            train_fn=train_fn,
            eval_fn=eval_fn,
            hyperparameter_space=self.hyperparameter_space,
            base_config=self.base_config,
            cv_folds=cv_folds
        )
        
        tuner = OptunaTuner(hpo_config, objective_fn, str(self.save_dir / "hpo"))
        hpo_results = tuner.optimize()
        
        # Phase 2: Architecture search (simplified)
        logger.info("Phase 2: Architecture optimization")
        arch_results = self._run_architecture_search(
            train_fn, eval_fn, hpo_results['best_params'], cv_folds
        )
        
        # Phase 3: Final model training and evaluation
        logger.info("Phase 3: Final model training")
        final_results = self._train_final_model(
            train_fn, eval_fn, arch_results['best_config']
        )
        
        # Combine all results
        automl_results = {
            'hpo_results': hpo_results,
            'architecture_results': arch_results,
            'final_results': final_results,
            'best_config': arch_results['best_config']
        }
        
        # Save AutoML results
        self._save_automl_results(automl_results)
        
        logger.info("AutoML pipeline completed")
        return automl_results
        
    def _run_architecture_search(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        best_hpo_params: Dict[str, Any],
        cv_folds: int
    ) -> Dict[str, Any]:
        """Run architecture search phase."""
        # This is a simplified architecture search
        # In practice, this could use more sophisticated methods like DARTS or ENAS
        
        architectures = [
            {'model_type': 'transformer', 'n_layers': 6, 'd_model': 256},
            {'model_type': 'transformer', 'n_layers': 8, 'd_model': 384},
            {'model_type': 'enhanced_transformer', 'n_layers': 6, 'd_model': 256},
            {'model_type': 'enhanced_transformer', 'n_layers': 8, 'd_model': 384},
        ]
        
        best_arch = None
        best_score = float('-inf')
        arch_results = []
        
        for arch in architectures:
            # Combine architecture with best HPO parameters
            config = self.base_config.copy()
            config.update(best_hpo_params)
            config.update(arch)
            
            # Run cross-validation
            cv_scores = []
            for fold in range(cv_folds):
                config['fold_idx'] = fold
                model, _ = train_fn(config)
                eval_metrics = eval_fn(model, config)
                score = eval_metrics.get('val_combined_score', eval_metrics.get('val_accuracy', 0.0))
                cv_scores.append(score)
                
            mean_score = np.mean(cv_scores)
            
            arch_result = {
                'architecture': arch,
                'cv_scores': cv_scores,
                'mean_score': mean_score,
                'std_score': np.std(cv_scores)
            }
            arch_results.append(arch_result)
            
            if mean_score > best_score:
                best_score = mean_score
                best_arch = arch
                
        final_config = self.base_config.copy()
        final_config.update(best_hpo_params)
        final_config.update(best_arch)
        
        return {
            'architectures_tested': arch_results,
            'best_architecture': best_arch,
            'best_score': best_score,
            'best_config': final_config
        }
        
    def _train_final_model(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        final_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train final model with best configuration."""
        # Train with full dataset and more epochs
        final_config['max_epochs'] = final_config.get('max_epochs', 100) * 2
        
        model, train_metrics = train_fn(final_config)
        eval_metrics = eval_fn(model, final_config)
        
        return {
            'final_config': final_config,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'model': model
        }
        
    def _save_automl_results(self, results: Dict[str, Any]):
        """Save AutoML results."""
        # Remove model object for JSON serialization
        results_for_json = results.copy()
        if 'model' in results_for_json['final_results']:
            del results_for_json['final_results']['model']
            
        # Save results
        results_path = self.save_dir / 'automl_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2, default=str)
            
        # Save best config
        config_path = self.save_dir / 'best_automl_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(results['best_config'], f, default_flow_style=False)


def create_hpo_study(
    study_name: str,
    storage: Optional[str] = None,
    sampler: str = "tpe",
    pruner: str = "median",
    direction: str = "maximize"
) -> optuna.Study:
    """
    Factory function to create Optuna study.
    
    Args:
        study_name: Name of the study
        storage: Storage URL for distributed optimization
        sampler: Sampling algorithm ("tpe", "random", "cmaes")
        pruner: Pruning algorithm ("median", "hyperband", "none")
        direction: Optimization direction ("maximize", "minimize")
        
    Returns:
        Configured Optuna study
    """
    # Configure sampler
    if sampler == "tpe":
        sampler_obj = TPESampler(seed=42)
    elif sampler == "random":
        sampler_obj = RandomSampler(seed=42)
    elif sampler == "cmaes":
        sampler_obj = CmaEsSampler(seed=42)
    else:
        sampler_obj = TPESampler(seed=42)
        
    # Configure pruner
    if pruner == "median":
        pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif pruner == "hyperband":
        pruner_obj = HyperbandPruner(min_resource=1, max_resource=50, reduction_factor=3)
    else:
        pruner_obj = optuna.pruners.NopPruner()
        
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler_obj,
        pruner=pruner_obj,
        direction=direction,
        load_if_exists=True
    )
    
    return study


def analyze_hpo_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze HPO results and provide insights.
    
    Args:
        results: HPO results dictionary
        
    Returns:
        Analysis results
    """
    analysis = {
        'optimization_summary': {},
        'parameter_insights': {},
        'convergence_analysis': {},
        'recommendations': []
    }
    
    # Optimization summary
    analysis['optimization_summary'] = {
        'total_trials': results['n_trials'],
        'optimization_time': results['optimization_time'],
        'best_value': results['best_value'],
        'trials_per_minute': results['n_trials'] / (results['optimization_time'] / 60) if results['optimization_time'] > 0 else 0
    }
    
    # Parameter importance analysis
    if 'parameter_importance' in results and results['parameter_importance']:
        sorted_importance = sorted(
            results['parameter_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        analysis['parameter_insights']['most_important'] = sorted_importance[:5]
        analysis['parameter_insights']['least_important'] = sorted_importance[-5:]
        
    # Convergence analysis
    if 'optimization_history' in results:
        values = results['optimization_history']
        if len(values) > 10:
            # Check if optimization is converging
            recent_values = values[-10:]
            early_values = values[:10]
            
            improvement = max(recent_values) - max(early_values)
            analysis['convergence_analysis']['improvement'] = improvement
            analysis['convergence_analysis']['converged'] = improvement < 0.01  # Threshold
            
            # Best value progression
            best_values = []
            current_best = float('-inf')
            for value in values:
                if value > current_best:
                    current_best = value
                best_values.append(current_best)
                
            analysis['convergence_analysis']['best_value_progression'] = best_values
            
    # Generate recommendations
    if analysis['convergence_analysis'].get('converged', False):
        analysis['recommendations'].append("Optimization appears to have converged. Consider stopping or trying different search space.")
    else:
        analysis['recommendations'].append("Optimization may benefit from more trials.")
        
    if 'parameter_importance' in results and results['parameter_importance']:
        top_param = max(results['parameter_importance'].items(), key=lambda x: x[1])
        analysis['recommendations'].append(f"Focus on tuning '{top_param[0]}' as it has the highest importance ({top_param[1]:.3f}).")
        
    return analysis


def extract_best_hyperparameters(study: optuna.Study) -> Dict[str, Any]:
    """
    Extract best hyperparameters from Optuna study.
    
    Args:
        study: Completed Optuna study
        
    Returns:
        Best hyperparameters
    """
    if study.best_trial is None:
        raise ValueError("No completed trials found in study")
        
    return study.best_trial.params


def compare_hpo_trials(
    study: optuna.Study,
    metric_name: str = "value",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Compare top performing trials from HPO study.
    
    Args:
        study: Optuna study
        metric_name: Metric to compare
        top_k: Number of top trials to compare
        
    Returns:
        List of top trial information
    """
    # Get completed trials sorted by value
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        return []
        
    # Sort by value (assuming maximization)
    sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)
    
    top_trials = []
    for i, trial in enumerate(sorted_trials[:top_k]):
        trial_info = {
            'rank': i + 1,
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'duration': trial.duration.total_seconds() if trial.duration else None
        }
        top_trials.append(trial_info)
        
    return top_trials
