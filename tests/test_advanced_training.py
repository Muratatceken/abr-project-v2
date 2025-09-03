"""
Comprehensive test suite for advanced training features in the ABR project.

Tests cover:
- Focal Loss implementation
- Knowledge Distillation
- Curriculum Learning
- Advanced Augmentations
- Early Stopping
- Ensemble Methods
- Cross-Validation
- Hyperparameter Optimization
- Training Monitoring
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
from utils.losses import FocalLoss, CombinedLoss, create_loss_from_config
from data.curriculum import (
    DifficultyMetrics, CurriculumScheduler, CurriculumSampler, 
    CurriculumDataset, create_curriculum_from_config
)
from data.augmentations import (
    MixUpAugmentation, CutMixAugmentation, AugmentationPipeline,
    create_augmentation_pipeline
)
from training.early_stopping import (
    EarlyStopping, MultiTaskEarlyStopping, AdaptiveEarlyStopping,
    EarlyStoppingCallback
)
from training.ensemble import (
    ModelEnsemble, SnapshotEnsemble, CrossValidationEnsemble,
    EnsemblePredictor
)
from training.distillation import (
    KnowledgeDistillation, MultiTaskDistillation, FeatureMatching,
    DistillationTrainer
)
from training.cross_validation import (
    CrossValidationManager, StratifiedPatientKFold,
    CrossValidationTrainer
)
from optimization.hyperparameter_optimization import (
    OptunaTuner, HyperparameterSpace, HPOObjective,
    AutoML
)
from training.monitoring import (
    TrainingMonitor, MetricTracker, ResourceMonitor,
    ModelHealthMonitor
)


class TestFocalLoss:
    """Test suite for FocalLoss implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.num_classes = 2
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization with different parameters."""
        # Test default parameters
        loss = FocalLoss()
        assert loss.alpha == 1.0
        assert loss.gamma == 2.0
        assert loss.reduction == 'mean'
        
        # Test custom parameters
        loss = FocalLoss(alpha=0.5, gamma=1.5, reduction='sum')
        assert loss.alpha == 0.5
        assert loss.gamma == 1.5
        assert loss.reduction == 'sum'
        
    def test_focal_loss_forward(self):
        """Test FocalLoss forward pass."""
        # Create sample data
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Compute loss
        loss = self.focal_loss(logits, targets)
        
        # Check output properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # non-negative
        
    def test_focal_loss_vs_ce_loss(self):
        """Test that focal loss reduces to CE loss when gamma=0."""
        # Create focal loss with gamma=0
        focal_loss_gamma0 = FocalLoss(alpha=1.0, gamma=0.0)
        ce_loss = nn.CrossEntropyLoss()
        
        # Create sample data
        logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Compute losses
        focal_result = focal_loss_gamma0(logits, targets)
        ce_result = ce_loss(logits, targets)
        
        # Should be approximately equal
        assert torch.allclose(focal_result, ce_result, atol=1e-5)
        
    def test_focal_loss_config_creation(self):
        """Test creating focal loss from configuration."""
        config = {
            'type': 'focal',
            'alpha': 0.25,
            'gamma': 2.0,
            'reduction': 'mean'
        }
        
        loss = create_loss_from_config(config)
        assert isinstance(loss, FocalLoss)
        assert loss.alpha == 0.25
        assert loss.gamma == 2.0


class TestDistillationLoss:
    """Test suite for DistillationLoss implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.num_classes = 10
        self.temperature = 4.0
        self.alpha = 0.7
        self.distill_loss = DistillationLoss(
            temperature=self.temperature,
            alpha=self.alpha
        )
        
    def test_distillation_loss_forward(self):
        """Test DistillationLoss forward pass."""
        # Create sample data
        student_logits = torch.randn(self.batch_size, self.num_classes)
        teacher_logits = torch.randn(self.batch_size, self.num_classes)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Compute loss
        loss = self.distill_loss(student_logits, teacher_logits, targets)
        
        # Check output properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # non-negative
        
    def test_feature_matching_loss(self):
        """Test feature matching component."""
        # Create sample features
        student_features = torch.randn(self.batch_size, 128)
        teacher_features = torch.randn(self.batch_size, 128)
        
        # Compute feature matching loss
        loss = self.distill_loss.feature_matching_loss(
            student_features, teacher_features
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestCurriculumLearning:
    """Test suite for Curriculum Learning implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.num_samples = 100
        self.difficulty_metrics = DifficultyMetrics()
        
        # Create mock dataset samples
        self.samples = []
        for i in range(self.num_samples):
            sample = {
                'signal': torch.randn(1000),
                'snr': np.random.uniform(-10, 30),
                'peak_count': np.random.randint(1, 8),
                'hearing_loss': np.random.uniform(0, 100),
                'metadata': {'patient_id': f'patient_{i % 20}'}
            }
            self.samples.append(sample)
            
    def test_difficulty_metrics_calculation(self):
        """Test difficulty metrics calculation."""
        sample = self.samples[0]
        
        # Test SNR difficulty
        snr_difficulty = self.difficulty_metrics.calculate_snr_difficulty(sample)
        assert 0 <= snr_difficulty <= 1
        
        # Test peak complexity
        peak_difficulty = self.difficulty_metrics.calculate_peak_complexity(sample)
        assert 0 <= peak_difficulty <= 1
        
        # Test hearing loss difficulty
        hearing_difficulty = self.difficulty_metrics.calculate_hearing_loss_difficulty(sample)
        assert 0 <= hearing_difficulty <= 1
        
        # Test combined difficulty
        combined_difficulty = self.difficulty_metrics.calculate_difficulty(sample)
        assert 0 <= combined_difficulty <= 1
        
    def test_curriculum_scheduler(self):
        """Test curriculum scheduler implementations."""
        total_epochs = 100
        
        # Test linear scheduler
        linear_scheduler = CurriculumScheduler.create_scheduler(
            'linear', total_epochs
        )
        
        # Test progression
        assert linear_scheduler.get_difficulty_threshold(0) == 0.0
        assert linear_scheduler.get_difficulty_threshold(total_epochs // 2) == 0.5
        assert linear_scheduler.get_difficulty_threshold(total_epochs) == 1.0
        
        # Test exponential scheduler
        exp_scheduler = CurriculumScheduler.create_scheduler(
            'exponential', total_epochs, decay_rate=0.05
        )
        
        threshold_0 = exp_scheduler.get_difficulty_threshold(0)
        threshold_half = exp_scheduler.get_difficulty_threshold(total_epochs // 2)
        threshold_end = exp_scheduler.get_difficulty_threshold(total_epochs)
        
        assert 0 <= threshold_0 < threshold_half < threshold_end <= 1
        
    def test_curriculum_dataset(self):
        """Test CurriculumDataset wrapper."""
        # Create mock base dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=self.num_samples)
        mock_dataset.__getitem__ = Mock(side_effect=lambda i: self.samples[i])
        
        # Create curriculum dataset
        curriculum_config = {
            'enabled': True,
            'scheduler_type': 'linear',
            'total_epochs': 100,
            'difficulty_weights': {
                'snr': 0.4,
                'peak_complexity': 0.3,
                'hearing_loss': 0.3
            }
        }
        
        curriculum_dataset = CurriculumDataset(
            mock_dataset, 
            curriculum_config,
            batch_size=16
        )
        
        # Test dataset properties
        assert len(curriculum_dataset) == self.num_samples
        
        # Test epoch update
        curriculum_dataset.update_epoch(10)
        assert curriculum_dataset.current_epoch == 10
        
        # Test sampler creation
        sampler = curriculum_dataset.create_sampler()
        assert sampler is not None


class TestAdvancedAugmentations:
    """Test suite for Advanced Augmentations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.signal_length = 1000
        self.num_classes = 2
        
    def test_mixup_augmentation(self):
        """Test MixUp augmentation."""
        mixup = MixUpAugmentation(alpha=1.0)
        
        # Create sample batch with ABR format
        batch = {
            'x0': torch.randn(self.batch_size, 1, self.signal_length),
            'peak_exists': torch.randint(0, 2, (self.batch_size,)).float(),
            'meta': [{'target': i % self.num_classes} for i in range(self.batch_size)]
        }
        
        # Apply mixup
        mixed_batch = mixup(batch)
        
        # Check output shapes
        assert mixed_batch['x0'].shape == batch['x0'].shape
        assert mixed_batch['peak_exists'].shape == batch['peak_exists'].shape
        
        # Check that signals are actually mixed (different from original)
        assert not torch.equal(mixed_batch['x0'], batch['x0'])
        
    def test_cutmix_augmentation(self):
        """Test CutMix augmentation."""
        cutmix = CutMixAugmentation(alpha=1.0)
        
        # Create sample batch with ABR format
        batch = {
            'x0': torch.randn(self.batch_size, 1, self.signal_length),
            'peak_exists': torch.randint(0, 2, (self.batch_size,)).float(),
            'meta': [{'target': i % self.num_classes} for i in range(self.batch_size)]
        }
        
        # Apply cutmix
        mixed_batch = cutmix(batch)
        
        # Check output shapes
        assert mixed_batch['x0'].shape == batch['x0'].shape
        assert mixed_batch['peak_exists'].shape == batch['peak_exists'].shape
        
        # Check that signals are actually mixed (different from original)
        assert not torch.equal(mixed_batch['x0'], batch['x0'])
        
    def test_augmentation_pipeline(self):
        """Test AugmentationPipeline with [1,T] per-sample format."""
        # Create real augmentations for pipeline
        from data.augmentations import ABRAugmentations
        
        augmentations = [
            (ABRAugmentations(apply_prob=1.0), 1.0),  # Always apply
        ]
        
        # Create pipeline
        pipeline = AugmentationPipeline(
            augmentations=augmentations,
            curriculum_aware=False
        )
        
        # Test with [1,T] per-sample format
        sample = {
            'x0': torch.randn(1, self.signal_length),  # [1, T] format
            'peak_exists': torch.tensor([1.0]),
            'meta': {'target': 0}
        }
        
        result = pipeline(sample)
        
        # Check that result maintains format and shape
        assert isinstance(result, dict)
        assert 'x0' in result
        assert result['x0'].shape == sample['x0'].shape


class TestEarlyStopping:
    """Test suite for Early Stopping implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
            mode='min'
        )
        
    def test_early_stopping_improvement(self):
        """Test early stopping with improving metrics."""
        # Simulate improving validation loss
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        
        for loss in losses:
            should_stop = self.early_stopping(loss)
            assert not should_stop
            
        # Check that best score is updated
        assert self.early_stopping.best_score == 0.6
        assert self.early_stopping.counter == 0
        
    def test_early_stopping_no_improvement(self):
        """Test early stopping when no improvement."""
        # Start with good loss, then no improvement
        self.early_stopping(0.5)  # Initial good loss
        
        # Now simulate no improvement
        for i in range(6):  # patience + 1
            should_stop = self.early_stopping(0.6)  # Worse loss
            if i < 4:  # Before patience is exceeded
                assert not should_stop
            else:  # After patience is exceeded
                assert should_stop
                break
                
    def test_multi_task_early_stopping(self):
        """Test MultiTaskEarlyStopping."""
        multi_early_stopping = MultiTaskEarlyStopping(
            metrics=['loss', 'peak_f1'],
            patience=3,
            mode={'loss': 'min', 'peak_f1': 'max'}
        )
        
        # Test with improving metrics
        metrics = {'loss': 0.5, 'peak_f1': 0.8}
        should_stop = multi_early_stopping(metrics)
        assert not should_stop
        
        # Test with mixed improvement
        metrics = {'loss': 0.6, 'peak_f1': 0.9}  # loss worse, f1 better
        should_stop = multi_early_stopping(metrics)
        assert not should_stop


class TestEnsembleMethods:
    """Test suite for Ensemble Methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.num_models = 3
        self.input_dim = 100
        self.output_dim = 10
        
        # Create mock models
        self.models = []
        for _ in range(self.num_models):
            model = Mock()
            model.eval = Mock()
            model.return_value = torch.randn(1, self.output_dim)
            self.models.append(model)
            
    def test_model_ensemble(self):
        """Test ModelEnsemble for prediction averaging."""
        ensemble = ModelEnsemble(self.models)
        
        # Test prediction
        input_data = torch.randn(1, self.input_dim)
        prediction = ensemble.predict(input_data)
        
        # Check output shape
        assert prediction.shape == (1, self.output_dim)
        
        # Check that all models were called
        for model in self.models:
            model.assert_called_once()
            
    def test_snapshot_ensemble(self):
        """Test SnapshotEnsemble for collecting model snapshots."""
        ensemble = SnapshotEnsemble(
            snapshot_epochs=[10, 20, 30],
            save_dir=tempfile.mkdtemp()
        )
        
        # Test snapshot decision
        assert not ensemble.should_take_snapshot(5)
        assert ensemble.should_take_snapshot(10)
        assert not ensemble.should_take_snapshot(15)
        
        # Test snapshot taking (mock)
        mock_model = Mock()
        mock_model.state_dict = Mock(return_value={'param': torch.tensor([1.0])})
        
        with patch('torch.save') as mock_save:
            ensemble.take_snapshot(mock_model, 10, {'loss': 0.5})
            mock_save.assert_called_once()
            
    def test_ensemble_predictor(self):
        """Test unified EnsemblePredictor interface."""
        predictor = EnsemblePredictor(self.models)
        
        # Test prediction with uncertainty
        input_data = torch.randn(1, self.input_dim)
        mean_pred, std_pred = predictor.predict_with_uncertainty(input_data)
        
        assert mean_pred.shape == (1, self.output_dim)
        assert std_pred.shape == (1, self.output_dim)
        assert torch.all(std_pred >= 0)  # Standard deviation should be non-negative


class TestKnowledgeDistillation:
    """Test suite for Knowledge Distillation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 100
        self.output_dim = 10
        
        # Create mock teacher model that accepts dict inputs
        class MockTeacherModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                
            def forward(self, x0, stat=None):
                return self.linear(x0.squeeze())
        
        self.teacher = MockTeacherModel(self.input_dim, self.output_dim)
        
        self.student = Mock()
        self.student.train = Mock()
        self.student.return_value = torch.randn(1, self.output_dim)
        
    def test_knowledge_distillation(self):
        """Test KnowledgeDistillation class."""
        distillation = KnowledgeDistillation(
            teacher=self.teacher,
            temperature=4.0,
            alpha=0.7
        )
        
        # Test distillation loss computation
        input_data = torch.randn(1, self.input_dim)
        student_logits = torch.randn(1, self.output_dim)
        targets = torch.randint(0, self.output_dim, (1,))
        
        loss = distillation.compute_loss(student_logits, input_data, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        
    def test_distillation_trainer(self):
        """Test DistillationTrainer."""
        # Create distillation method
        distillation_method = KnowledgeDistillation(
            temperature=4.0,
            alpha=0.7,
            feature_matching=True
        )
        
        trainer = DistillationTrainer(
            distillation_method=distillation_method,
            teacher_model=self.teacher
        )
        
        # Test training step with proper batch format
        batch = {
            'x0': torch.randn(1, self.input_dim),
            'peak_exists': torch.randint(0, 2, (1,)).float(),
            'stat': None
        }
        
        # Create a mock student model that accepts dict inputs
        class MockStudentModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                
            def forward(self, x0, stat=None):
                return self.linear(x0.squeeze())
        
        student_model = MockStudentModel(self.input_dim, self.output_dim)
        
        distill_losses = trainer.compute_distillation_loss(
            student_model=student_model,
            batch=batch,
            epoch=0
        )
        
        # Check returned dict keys
        assert isinstance(distill_losses, dict), "Should return dict of losses"
        assert 'distillation_loss' in distill_losses, "Should contain distillation_loss key"
        assert isinstance(distill_losses['distillation_loss'], torch.Tensor)
        assert distill_losses['distillation_loss'].item() >= 0


class TestCrossValidation:
    """Test suite for Cross-Validation framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.num_samples = 100
        self.num_patients = 20
        
        # Create mock data with patient groups
        self.data = []
        self.patients = []
        for i in range(self.num_samples):
            patient_id = f"patient_{i % self.num_patients}"
            self.data.append({
                'x0': torch.randn(1, 1000),  # [1, T] format
                'peak_exists': torch.randint(0, 2, (1,)).float(),
                'meta': {'target': i % 2}  # Class index
            })
            self.patients.append(patient_id)
            
    def test_stratified_patient_kfold(self):
        """Test StratifiedPatientKFold splitter."""
        # Create mock labels
        labels = np.random.randint(0, 2, self.num_samples)
        
        splitter = StratifiedPatientKFold(
            n_splits=5,
            patient_groups=self.patients
        )
        
        splits = list(splitter.split(self.data, labels))
        
        # Check number of splits
        assert len(splits) == 5
        
        # Check that each split has train and validation indices
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert len(set(train_idx) & set(val_idx)) == 0  # No overlap
            
    def test_cross_validation_manager(self):
        """Test CrossValidationManager."""
        cv_config = {
            'n_splits': 3,
            'stratified': True,
            'patient_stratified': True
        }
        
        manager = CrossValidationManager(cv_config)
        
        # Test split creation
        splits = manager.create_splits(self.data, patient_groups=self.patients)
        assert len(splits) == 3
        
    def test_cross_validation_trainer(self):
        """Test CrossValidationTrainer."""
        # Create temporary directory for CV results
        temp_dir = tempfile.mkdtemp()
        
        try:
            cv_config = {
                'n_splits': 2,
                'save_models': True,
                'results_dir': temp_dir
            }
            
            trainer = CrossValidationTrainer(cv_config)
            
            # Mock training function
            def mock_train_function(train_data, val_data, fold):
                return {
                    'model': Mock(),
                    'metrics': {'loss': 0.5, 'accuracy': 0.8}
                }
            
            # Test CV training
            results = trainer.run_cross_validation(
                data=self.data,
                train_function=mock_train_function,
                patient_groups=self.patients
            )
            
            assert 'fold_results' in results
            assert 'aggregated_metrics' in results
            assert len(results['fold_results']) == 2
            
        finally:
            shutil.rmtree(temp_dir)


class TestHyperparameterOptimization:
    """Test suite for Hyperparameter Optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = {
            'learning_rate': ('float', 1e-5, 1e-2, 'log'),
            'batch_size': ('int', 16, 128),
            'hidden_dim': ('categorical', [128, 256, 512])
        }
        
    def test_hyperparameter_space(self):
        """Test HyperparameterSpace creation."""
        space = HyperparameterSpace(self.search_space)
        
        # Test space definition
        assert 'learning_rate' in space.space
        assert 'batch_size' in space.space
        assert 'hidden_dim' in space.space
        
        # Mock trial for sampling
        mock_trial = Mock()
        mock_trial.suggest_float = Mock(return_value=0.001)
        mock_trial.suggest_int = Mock(return_value=64)
        mock_trial.suggest_categorical = Mock(return_value=256)
        
        params = space.sample(mock_trial)
        
        assert 'learning_rate' in params
        assert 'batch_size' in params
        assert 'hidden_dim' in params
        
    @patch('optuna.create_study')
    def test_optuna_tuner(self, mock_create_study):
        """Test OptunaTuner."""
        # Mock study
        mock_study = Mock()
        mock_study.optimize = Mock()
        mock_study.best_params = {'learning_rate': 0.001}
        mock_study.best_value = 0.5
        mock_create_study.return_value = mock_study
        
        tuner = OptunaTuner(
            search_space=self.search_space,
            direction='minimize'
        )
        
        # Mock objective function
        def mock_objective(params):
            return 0.5
        
        # Test optimization
        best_params = tuner.optimize(
            objective=mock_objective,
            n_trials=10
        )
        
        assert best_params is not None
        mock_study.optimize.assert_called_once()
        
    def test_hpo_trainer(self):
        """Test HPOTrainer wrapper."""
        trainer = HPOTrainer(
            base_config={'model': 'transformer'},
            search_space=self.search_space
        )
        
        # Mock training function
        def mock_train_function(config):
            return {'loss': 0.5, 'accuracy': 0.8}
        
        # Test parameter merging
        trial_params = {'learning_rate': 0.001}
        merged_config = trainer._merge_params(trial_params)
        
        assert 'model' in merged_config
        assert 'learning_rate' in merged_config


class TestTrainingMonitoring:
    """Test suite for Training Monitoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_metric_tracker(self):
        """Test MetricTracker."""
        tracker = MetricTracker()
        
        # Add metrics
        tracker.update('loss', 0.5, step=1)
        tracker.update('accuracy', 0.8, step=1)
        tracker.update('loss', 0.4, step=2)
        
        # Test retrieval
        loss_history = tracker.get_metric_history('loss')
        assert len(loss_history) == 2
        assert loss_history[0] == (1, 0.5)
        assert loss_history[1] == (2, 0.4)
        
        # Test latest values
        latest = tracker.get_latest_metrics()
        assert 'loss' in latest
        assert 'accuracy' in latest
        
    def test_resource_monitor(self):
        """Test ResourceMonitor."""
        monitor = ResourceMonitor()
        
        # Test resource collection
        resources = monitor.get_current_resources()
        
        assert 'cpu_percent' in resources
        assert 'memory_percent' in resources
        assert 'gpu_memory_used' in resources
        
        # Check that values are reasonable
        assert 0 <= resources['cpu_percent'] <= 100
        assert 0 <= resources['memory_percent'] <= 100
        
    def test_model_health_monitor(self):
        """Test ModelHealthMonitor."""
        monitor = ModelHealthMonitor()
        
        # Create mock model with parameters
        model = nn.Linear(10, 1)
        
        # Test gradient monitoring
        # First, create some gradients
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        health_metrics = monitor.check_model_health(model)
        
        assert 'gradient_norm' in health_metrics
        assert 'parameter_norm' in health_metrics
        assert health_metrics['gradient_norm'] >= 0
        assert health_metrics['parameter_norm'] >= 0
        
    def test_training_monitor_integration(self):
        """Test TrainingMonitor integration."""
        monitor = TrainingMonitor(
            log_dir=self.temp_dir,
            log_frequency=1
        )
        
        # Test metric update
        metrics = {
            'loss': 0.5,
            'accuracy': 0.8,
            'learning_rate': 0.001
        }
        
        monitor.update(metrics, step=1, epoch=1)
        
        # Test that metrics are stored
        latest = monitor.metric_tracker.get_latest_metrics()
        assert 'loss' in latest
        assert 'accuracy' in latest
        
        # Test report generation
        report = monitor.generate_report()
        assert 'training_summary' in report
        assert 'resource_usage' in report


class TestConfigurationIntegration:
    """Test configuration-based creation and integration."""
    
    def test_loss_from_config(self):
        """Test creating losses from configuration."""
        # Test focal loss config
        focal_config = {
            'type': 'focal',
            'alpha': 0.25,
            'gamma': 2.0
        }
        
        loss = create_loss_from_config(focal_config)
        assert isinstance(loss, FocalLoss)
        
        # Test combined loss config
        combined_config = {
            'type': 'combined',
            'losses': {
                'reconstruction': {'type': 'mse', 'weight': 1.0},
                'peak_classification': {'type': 'focal', 'alpha': 0.25, 'weight': 0.5}
            }
        }
        
        loss = create_loss_from_config(combined_config)
        assert isinstance(loss, CombinedLoss)
        
    def test_curriculum_from_config(self):
        """Test creating curriculum learning from configuration."""
        config = {
            'enabled': True,
            'scheduler_type': 'linear',
            'total_epochs': 100,
            'difficulty_weights': {
                'snr': 0.5,
                'peak_complexity': 0.3,
                'hearing_loss': 0.2
            }
        }
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        curriculum_dataset = create_curriculum_from_config(
            mock_dataset, config, batch_size=32
        )
        
        assert curriculum_dataset is not None
        
    def test_augmentation_pipeline_from_config(self):
        """Test creating augmentation pipeline from configuration."""
        config = {
            'enabled': True,
            'mixup_prob': 0.5,
            'cutmix_prob': 0.5,
            'augmentation_strength': 0.8,
            'curriculum_aware': True,
            'pipeline': [
                {'type': 'time_shift', 'max_shift': 0.1},
                {'type': 'amplitude_scale', 'scale_range': [0.8, 1.2]}
            ]
        }
        
        pipeline = create_augmentation_pipeline(config)
        assert pipeline is not None


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_advanced_training_pipeline(self):
        """Test complete advanced training pipeline integration."""
        # Create comprehensive config
        config = {
            'focal_loss': {
                'enabled': True,
                'alpha': 0.25,
                'gamma': 2.0
            },
            'curriculum_learning': {
                'enabled': True,
                'scheduler_type': 'linear',
                'total_epochs': 10
            },
            'early_stopping': {
                'enabled': True,
                'patience': 3,
                'min_delta': 0.001
            },
            'monitoring': {
                'enabled': True,
                'log_dir': self.temp_dir
            }
        }
        
        # Mock components
        mock_model = nn.Linear(100, 10)
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset.__getitem__ = Mock(return_value={
            'signal': torch.randn(100),
            'target': torch.randint(0, 10, (1,)),
            'snr': 10.0,
            'peak_count': 3,
            'hearing_loss': 20.0
        })
        
        # Test component creation
        focal_loss = create_loss_from_config(config['focal_loss'])
        assert isinstance(focal_loss, FocalLoss)
        
        curriculum_dataset = create_curriculum_from_config(
            mock_dataset, config['curriculum_learning'], batch_size=16
        )
        assert curriculum_dataset is not None
        
        early_stopping = EarlyStopping(
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta']
        )
        assert early_stopping is not None
        
        monitor = TrainingMonitor(
            log_dir=config['monitoring']['log_dir']
        )
        assert monitor is not None
        
        # Test integration
        # Simulate a few training steps
        for epoch in range(5):
            curriculum_dataset.update_epoch(epoch)
            
            # Simulate validation
            val_loss = 0.5 - epoch * 0.05  # Improving loss
            should_stop = early_stopping(val_loss)
            
            monitor.update({
                'loss': val_loss,
                'epoch': epoch
            }, step=epoch, epoch=epoch)
            
            if should_stop:
                break
                
        # Verify that monitoring worked
        report = monitor.generate_report()
        assert report is not None


if __name__ == '__main__':
    pytest.main([__file__])
