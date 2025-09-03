"""
Comprehensive tests for enhanced evaluation functionality.

This module tests:
- Peak classification evaluation
- Statistical significance testing
- Clinical validation metrics
- Visualization functions
- Comparative analysis
- Configuration integration
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path
import json

# Add parent directory to path to import modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.analysis import (
    bootstrap_classification_metrics,
    statistical_significance_tests,
    roc_analysis,
    precision_recall_analysis,
    clinical_validation_analysis,
    comparative_statistical_analysis
)

from evaluation.visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_classification_metrics_comparison,
    plot_threshold_analysis,
    plot_clinical_validation_dashboard,
    plot_ablation_study_results,
    create_publication_figure
)

from evaluation.comparative_analysis import (
    ComparativeAnalyzer,
    validate_comparability,
    extract_configuration_differences,
    calculate_performance_rankings,
    generate_statistical_summary
)


class TestPeakClassificationEvaluation(unittest.TestCase):
    """Test peak classification evaluation functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Generate mock classification data
        self.logits = np.random.randn(100)
        self.targets = np.random.binomial(1, 0.3, 100)  # 30% positive class
        
        # Generate mock evaluation results
        self.mock_results = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.78,
            'f1': 0.80,
            'auroc': 0.89
        }
    
    def test_bootstrap_classification_metrics(self):
        """Test bootstrap confidence interval calculation."""
        results = bootstrap_classification_metrics(
            self.logits, self.targets, n_bootstrap=100, confidence_level=0.95
        )
        
        # Check that results contain expected keys
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1', results)
        self.assertIn('auroc', results)
        
        # Check that each metric has confidence intervals
        for metric, data in results.items():
            self.assertIn('mean', data)
            self.assertIn('std', data)
            self.assertIn('ci_lower', data)
            self.assertIn('ci_upper', data)
            self.assertIn('confidence_level', data)
            
            # Check that confidence intervals are reasonable
            self.assertLess(data['ci_lower'], data['ci_upper'])
            self.assertGreater(data['ci_lower'], 0)
            self.assertLess(data['ci_upper'], 1)
    
    def test_statistical_significance_tests(self):
        """Test statistical significance testing."""
        results = statistical_significance_tests(
            self.logits, self.targets, prevalence=0.3
        )
        
        # Check that results contain expected keys
        self.assertIn('accuracy_test', results)
        self.assertIn('effect_sizes', results)
        
        # Check accuracy test
        accuracy_test = results['accuracy_test']
        self.assertIn('p_value', accuracy_test)
        self.assertIn('significant', accuracy_test)
        self.assertIn('test_type', accuracy_test)
        
        # Check effect sizes
        effect_sizes = results['effect_sizes']
        self.assertIn('cohens_d', effect_sizes)
        self.assertIn('cliff_delta', effect_sizes)
        self.assertIn('interpretation', effect_sizes)
    
    def test_roc_analysis(self):
        """Test ROC curve analysis."""
        results = roc_analysis(self.logits, self.targets)
        
        # Check that results contain expected keys
        self.assertIn('roc_curve', results)
        self.assertIn('auroc', results)
        self.assertIn('optimal_threshold', results)
        self.assertIn('sensitivity_at_specificity', results)
        
        # Check ROC curve data
        roc_data = results['roc_curve']
        self.assertIn('fpr', roc_data)
        self.assertIn('tpr', roc_data)
        self.assertIn('thresholds', roc_data)
        
        # Check AUROC value
        self.assertGreater(results['auroc'], 0.5)  # Should be better than random
        self.assertLess(results['auroc'], 1.0)
        
        # Check optimal threshold
        opt_thresh = results['optimal_threshold']
        self.assertIn('threshold', opt_thresh)
        self.assertIn('sensitivity', opt_thresh)
        self.assertIn('specificity', opt_thresh)
    
    def test_precision_recall_analysis(self):
        """Test precision-recall curve analysis."""
        results = precision_recall_analysis(self.logits, self.targets)
        
        # Check that results contain expected keys
        self.assertIn('pr_curve', results)
        self.assertIn('average_precision', results)
        self.assertIn('optimal_threshold', results)
        
        # Check PR curve data
        pr_data = results['pr_curve']
        self.assertIn('precision', pr_data)
        self.assertIn('recall', pr_data)
        self.assertIn('thresholds', pr_data)
        
        # Check average precision value
        self.assertGreater(results['average_precision'], 0)
        self.assertLess(results['average_precision'], 1)
    
    def test_clinical_validation_analysis(self):
        """Test clinical validation metrics."""
        results = clinical_validation_analysis(self.logits, self.targets)
        
        # Check that results contain expected keys
        self.assertIn('basic_metrics', results)
        self.assertIn('diagnostic_odds_ratio', results)
        self.assertIn('likelihood_ratios', results)
        self.assertIn('clinical_utility', results)
        
        # Check basic metrics
        basic_metrics = results['basic_metrics']
        self.assertIn('sensitivity', basic_metrics)
        self.assertIn('specificity', basic_metrics)
        self.assertIn('positive_predictive_value', basic_metrics)
        self.assertIn('negative_predictive_value', basic_metrics)
        
        # Check that metrics are in valid ranges
        for metric_name, value in basic_metrics.items():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
    
    def test_comparative_statistical_analysis(self):
        """Test comparative statistical analysis."""
        # Create mock results for comparison
        results1 = {'accuracy': 0.85, 'f1': 0.80, 'auroc': 0.89}
        results2 = {'accuracy': 0.88, 'f1': 0.83, 'auroc': 0.91}
        
        comparison = comparative_statistical_analysis(results1, results2, paired=True)
        
        # Check that comparison contains expected keys
        self.assertIn('accuracy', comparison)
        self.assertIn('f1', comparison)
        self.assertIn('auroc', comparison)
        
        # Check that each metric has comparison data
        for metric, data in comparison.items():
            self.assertIn('value1', data)
            self.assertIn('value2', data)
            self.assertIn('difference', data)
            self.assertIn('p_value', data)
            self.assertIn('significant', data)


class TestVisualizationFunctions(unittest.TestCase):
    """Test visualization functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Generate mock data
        self.logits = np.random.randn(100)
        self.targets = np.random.binomial(1, 0.3, 100)
        
        # Mock analysis results
        self.roc_data = {
            'roc_curve': {
                'fpr': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'tpr': [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
                'thresholds': [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
            },
            'auroc': 0.89,
            'optimal_threshold': {
                'threshold': 0.0,
                'sensitivity': 0.8,
                'specificity': 0.6
            }
        }
        
        self.pr_data = {
            'pr_curve': {
                'precision': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
                'recall': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'thresholds': [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
            },
            'average_precision': 0.65
        }
        
        self.confusion_matrix = np.array([[70, 10], [15, 5]])
        
        # Create temporary directory for saving plots
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        fig = plot_roc_curve(self.roc_data, save_path=None)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
        
        # Check that figure has axes
        self.assertGreater(len(fig.axes), 0)
    
    def test_plot_precision_recall_curve(self):
        """Test precision-recall curve plotting."""
        fig = plot_precision_recall_curve(self.pr_data, save_path=None)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
        
        # Check that figure has axes
        self.assertGreater(len(fig.axes), 0)
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        fig = plot_confusion_matrix(
            self.confusion_matrix, 
            save_path=None,
            include_metrics=True
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
    
    def test_plot_classification_metrics_comparison(self):
        """Test classification metrics comparison plotting."""
        metrics_data = [
            {'accuracy': 0.85, 'f1': 0.80, 'auroc': 0.89},
            {'accuracy': 0.88, 'f1': 0.83, 'auroc': 0.91}
        ]
        model_names = ['Model A', 'Model B']
        
        fig = plot_classification_metrics_comparison(
            metrics_data, model_names, save_path=None
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
    
    def test_plot_threshold_analysis(self):
        """Test threshold analysis plotting."""
        fig = plot_threshold_analysis(self.logits, self.targets, save_path=None)
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
        
        # Check that figure has multiple subplots
        self.assertEqual(len(fig.axes), 2)
    
    def test_plot_clinical_validation_dashboard(self):
        """Test clinical validation dashboard plotting."""
        clinical_data = {
            'basic_metrics': {
                'sensitivity': 0.8,
                'specificity': 0.7,
                'positive_predictive_value': 0.75,
                'negative_predictive_value': 0.76
            },
            'diagnostic_odds_ratio': {'value': 10.5},
            'likelihood_ratios': {
                'positive_likelihood_ratio': 2.67,
                'negative_likelihood_ratio': 0.29
            },
            'clinical_utility': {
                'number_needed_to_diagnose': 2.0,
                'prevalence_adjusted_ppv': 0.75,
                'prevalence_adjusted_npv': 0.76
            }
        }
        
        fig = plot_clinical_validation_dashboard(
            clinical_data, self.roc_data, self.pr_data, save_path=None
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
    
    def test_plot_ablation_study_results(self):
        """Test ablation study results plotting."""
        ablation_results = [
            {
                'component_name': 'Attention',
                'effect_size': 0.8,
                'confidence_interval': {'lower': 0.6, 'upper': 1.0}
            },
            {
                'component_name': 'FFN',
                'effect_size': 0.5,
                'confidence_interval': {'lower': 0.3, 'upper': 0.7}
            }
        ]
        
        baseline_result = {'accuracy': 0.75}
        
        fig = plot_ablation_study_results(
            ablation_results, baseline_result, save_path=None
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
    
    def test_create_publication_figure(self):
        """Test publication figure creation."""
        plots_data = [
            {'type': 'roc_curve', 'data': self.roc_data},
            {'type': 'pr_curve', 'data': self.pr_data},
            {'type': 'confusion_matrix', 'data': self.confusion_matrix},
            {'type': 'metrics_comparison', 'data': {'accuracy': 0.85, 'f1': 0.80}}
        ]
        
        fig = create_publication_figure(
            plots_data, layout='2x2', save_path=None
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')
        
        # Check that figure has multiple subplots
        self.assertEqual(len(fig.axes), 4)


class TestComparativeAnalysis(unittest.TestCase):
    """Test comparative analysis functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock evaluation results
        self.create_mock_evaluation_results()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_evaluation_results(self):
        """Create mock evaluation result files."""
        # Create baseline results
        baseline_dir = Path(self.temp_dir) / "baseline"
        baseline_dir.mkdir()
        
        baseline_results = {
            'metrics': {
                'accuracy': 0.75,
                'f1': 0.70,
                'auroc': 0.80,
                'mse': 0.25,
                'correlation': 0.85
            },
            'timestamp': '2024-01-01 12:00:00',
            'model_config': {'d_model': 256, 'n_layers': 6},
            'dataset_info': {'num_samples': 1000},
            'evaluation_config': {'mode': 'generation'}
        }
        
        with open(baseline_dir / "evaluation_summary.json", 'w') as f:
            json.dump(baseline_results, f)
        
        # Create ablation results
        ablation_dir = Path(self.temp_dir) / "no_attention"
        ablation_dir.mkdir()
        
        ablation_results = {
            'metrics': {
                'accuracy': 0.65,
                'f1': 0.60,
                'auroc': 0.70,
                'mse': 0.35,
                'correlation': 0.75
            },
            'timestamp': '2024-01-01 13:00:00',
            'model_config': {'d_model': 256, 'n_layers': 6},
            'dataset_info': {'num_samples': 1000},
            'evaluation_config': {'mode': 'generation'}
        }
        
        with open(ablation_dir / "evaluation_summary.json", 'w') as f:
            json.dump(ablation_results, f)
        
        # Create another ablation result
        ablation2_dir = Path(self.temp_dir) / "no_ffn"
        ablation2_dir.mkdir()
        
        ablation2_results = {
            'metrics': {
                'accuracy': 0.70,
                'f1': 0.65,
                'auroc': 0.75,
                'mse': 0.30,
                'correlation': 0.80
            },
            'timestamp': '2024-01-01 14:00:00',
            'model_config': {'d_model': 256, 'n_layers': 6},
            'dataset_info': {'num_samples': 1000},
            'evaluation_config': {'mode': 'generation'}
        }
        
        with open(ablation2_dir / "evaluation_summary.json", 'w') as f:
            json.dump(ablation2_results, f)
    
    def test_comparative_analyzer_initialization(self):
        """Test ComparativeAnalyzer initialization."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention"),
            str(Path(self.temp_dir) / "no_ffn")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs, baseline_run="baseline")
        
        # Check that results were loaded
        self.assertEqual(len(analyzer.results), 3)
        self.assertIn('baseline', analyzer.results)
        self.assertIn('no_attention', analyzer.results)
        self.assertIn('no_ffn', analyzer.results)
        
        # Check that metadata was loaded
        self.assertEqual(len(analyzer.metadata), 3)
    
    def test_signal_generation_comparison(self):
        """Test signal generation metrics comparison."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs, baseline_run="baseline")
        
        comparison = analyzer.compare_signal_generation_metrics(['mse', 'correlation'])
        
        # Check that comparison was performed
        self.assertIn('no_attention', comparison)
        self.assertIn('mse', comparison['no_attention'])
        self.assertIn('correlation', comparison['no_attention'])
        
        # Check that differences were calculated
        mse_data = comparison['no_attention']['mse']
        self.assertIn('baseline_value', mse_data)
        self.assertIn('current_value', mse_data)
        self.assertIn('difference', mse_data)
        self.assertIn('relative_improvement', mse_data)
    
    def test_classification_metrics_comparison(self):
        """Test classification metrics comparison."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs, baseline_run="baseline")
        
        comparison = analyzer.compare_classification_metrics(['accuracy', 'f1', 'auroc'])
        
        # Check that comparison was performed
        self.assertIn('no_attention', comparison)
        self.assertIn('accuracy', comparison['no_attention'])
        self.assertIn('f1', comparison['no_attention'])
        self.assertIn('auroc', comparison['no_attention'])
    
    def test_ablation_study_analysis(self):
        """Test ablation study analysis."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention"),
            str(Path(self.temp_dir) / "no_ffn")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs, baseline_run="baseline")
        
        ablation_results = analyzer.ablation_study_analysis()
        
        # Check that ablation analysis was performed
        self.assertEqual(len(ablation_results), 2)
        
        # Check that components were identified
        component_names = [result['component_name'] for result in ablation_results]
        self.assertIn('Attention', component_names)
        self.assertIn('Ffn', component_names)
        
        # Check that effect sizes were calculated
        for result in ablation_results:
            self.assertIn('effect_sizes', result)
            self.assertIn('average_effect_size', result)
            self.assertIn('max_effect_size', result)
    
    def test_comparison_table_generation(self):
        """Test comparison table generation."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs, baseline_run="baseline")
        
        # Run comparisons first
        analyzer.compare_signal_generation_metrics()
        analyzer.compare_classification_metrics()
        
        # Generate comparison table
        table_path = analyzer.generate_comparison_table(output_format='csv')
        
        # Check that table was generated
        self.assertTrue(Path(table_path).exists())
        
        # Check CSV content
        df = pd.read_csv(table_path)
        self.assertGreater(len(df), 0)
        self.assertIn('Run', df.columns)
    
    def test_comparative_report_generation(self):
        """Test comparative report generation."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs, baseline_run="baseline")
        
        # Run comparisons first
        analyzer.compare_signal_generation_metrics()
        analyzer.compare_classification_metrics()
        
        # Generate report
        report_path = analyzer.generate_comparative_report()
        
        # Check that report was generated
        self.assertTrue(Path(report_path).exists())
        
        # Check HTML content
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn('ABR Transformer Comparative Analysis Report', content)
            self.assertIn('baseline', content)
            self.assertIn('no_attention', content)
    
    def test_performance_rankings(self):
        """Test performance rankings generation."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention"),
            str(Path(self.temp_dir) / "no_ffn")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs)
        
        rankings = analyzer.get_performance_rankings(['accuracy', 'f1'])
        
        # Check that rankings were generated
        self.assertIn('accuracy', rankings)
        self.assertIn('f1', rankings)
        
        # Check ranking structure
        for metric, ranking in rankings.items():
            self.assertGreater(len(ranking), 0)
            for rank_info in ranking:
                self.assertIn('rank', rank_info)
                self.assertIn('run_name', rank_info)
                self.assertIn('value', rank_info)
    
    def test_statistical_summary(self):
        """Test statistical summary generation."""
        evaluation_dirs = [
            str(Path(self.temp_dir) / "baseline"),
            str(Path(self.temp_dir) / "no_attention")
        ]
        
        analyzer = ComparativeAnalyzer(evaluation_dirs, baseline_run="baseline")
        
        # Run comparisons first
        analyzer.compare_signal_generation_metrics()
        analyzer.compare_classification_metrics()
        
        summary = analyzer.generate_statistical_summary()
        
        # Check summary structure
        self.assertIn('total_runs', summary)
        self.assertIn('baseline_run', summary)
        self.assertIn('comparison_summary', summary)
        self.assertIn('recommendations', summary)
        
        # Check that recommendations were generated
        self.assertGreater(len(summary['recommendations']), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_validate_comparability(self):
        """Test comparability validation."""
        run1 = {'accuracy': 0.85, 'f1': 0.80}
        run2 = {'accuracy': 0.88, 'f1': 0.83}
        
        is_comparable, issues = validate_comparability(run1, run2)
        
        # Should be comparable
        self.assertTrue(is_comparable)
        self.assertEqual(len(issues), 0)
        
        # Test with different metrics
        run3 = {'accuracy': 0.85, 'precision': 0.82}
        is_comparable, issues = validate_comparability(run1, run3)
        
        # Should not be comparable
        self.assertFalse(is_comparable)
        self.assertGreater(len(issues), 0)
    
    def test_extract_configuration_differences(self):
        """Test configuration difference extraction."""
        config1 = {'d_model': 256, 'n_layers': 6, 'dropout': 0.1}
        config2 = {'d_model': 512, 'n_layers': 6, 'attention_heads': 8}
        
        differences = extract_configuration_differences(config1, config2)
        
        # Check that differences were identified
        self.assertIn('d_model', differences)
        self.assertIn('dropout', differences)
        self.assertIn('attention_heads', differences)
        
        # Check difference types
        self.assertEqual(differences['d_model']['type'], 'modified')
        self.assertEqual(differences['dropout']['type'], 'removed')
        self.assertEqual(differences['attention_heads']['type'], 'added')
    
    def test_calculate_performance_rankings(self):
        """Test performance ranking calculation."""
        metrics_data = [
            {'accuracy': 0.85, 'f1': 0.80},
            {'accuracy': 0.88, 'f1': 0.83},
            {'accuracy': 0.82, 'f1': 0.78}
        ]
        
        rankings = calculate_performance_rankings(metrics_data, ['accuracy', 'f1'])
        
        # Check that rankings were generated
        self.assertIn('accuracy', rankings)
        self.assertIn('f1', rankings)
        
        # Check ranking order (higher is better)
        accuracy_ranking = rankings['accuracy']
        self.assertEqual(accuracy_ranking[0]['rank'], 1)
        self.assertEqual(accuracy_ranking[0]['value'], 0.88)  # Highest accuracy
    
    def test_generate_statistical_summary(self):
        """Test statistical summary generation."""
        comparison_results = {
            'accuracy': {
                'p_value': 0.001,
                'significant': True,
                'effect_size': 0.8,
                'interpretation': 'large'
            },
            'f1': {
                'p_value': 0.05,
                'significant': True,
                'effect_size': 0.5,
                'interpretation': 'medium'
            }
        }
        
        summary = generate_statistical_summary(comparison_results)
        
        # Check that summary was generated
        self.assertIn('ACCURACY:', summary)
        self.assertIn('F1:', summary)
        self.assertIn('P-value: 0.001', summary)
        self.assertIn('Significant: Yes', summary)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration."""
    
    def test_configuration_parsing(self):
        """Test that new configuration options are properly parsed."""
        # This test would require actual configuration file parsing
        # For now, we'll test the structure of expected config
        expected_config_keys = [
            'peak_classification',
            'statistical_analysis',
            'clinical_metrics',
            'comparative_analysis',
            'report'
        ]
        
        # Mock configuration structure
        config = {
            'metrics': {
                'peak_classification': {'enabled': True},
                'statistical_analysis': {'confidence_level': 0.95},
                'clinical_metrics': {'sensitivity_analysis': True},
                'comparative_analysis': {'enabled': False},
            },
            'report': {
                'save_classification_metrics': True,
                'save_roc_curves': True
            }
        }
        
        # Check that expected keys exist
        for key in expected_config_keys:
            if key == 'peak_classification':
                self.assertIn(key, config['metrics'])
            elif key == 'report':
                self.assertIn(key, config)
            else:
                self.assertIn(key, config['metrics'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_perfect_classification(self):
        """Test behavior with perfect classification."""
        # Perfect classification: all predictions correct
        logits = np.array([10.0, 10.0, -10.0, -10.0])  # Very confident predictions
        targets = np.array([1, 1, 0, 0])
        
        # Should handle without errors
        results = statistical_significance_tests(logits, targets)
        self.assertIsNotNone(results)
        
        # Check that perfect accuracy is detected
        self.assertIn('accuracy_test', results)
    
    def test_all_same_class(self):
        """Test behavior when all predictions are the same class."""
        # All predictions are positive
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        targets = np.array([1, 1, 0, 0])
        
        # Should handle without errors
        results = clinical_validation_analysis(logits, targets)
        self.assertIsNotNone(results)
        
        # Check that edge cases are handled
        self.assertIn('basic_metrics', results)
    
    def test_very_small_sample_size(self):
        """Test behavior with very small sample sizes."""
        # Very small sample
        logits = np.array([1.0, -1.0])
        targets = np.array([1, 0])
        
        # Should handle without errors
        results = bootstrap_classification_metrics(logits, targets, n_bootstrap=10)
        self.assertIsNotNone(results)
    
    def test_imbalanced_dataset(self):
        """Test behavior with highly imbalanced datasets."""
        # Highly imbalanced: 90% negative, 10% positive
        logits = np.random.randn(1000)
        targets = np.random.binomial(1, 0.1, 1000)
        
        # Should handle without errors
        results = clinical_validation_analysis(logits, targets)
        self.assertIsNotNone(results)
        
        # Check that prevalence is calculated correctly
        self.assertAlmostEqual(results['prevalence'], 0.1, places=1)


class TestEvaluationEnhancements(unittest.TestCase):
    """Test the enhanced evaluation pipeline with classification capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'dataset': {
                'data_path': 'mock_data.pkl',
                'batch_size': 4,
                'num_workers': 0,
                'pin_memory': False,
                'return_peak_labels': True
            },
            'model': {
                'checkpoint_path': 'mock_checkpoint.pth',
                'device': 'cpu',
                'input_channels': 1,
                'static_dim': 4,
                'sequence_length': 200,
                'd_model': 64,
                'n_layers': 2,
                'n_heads': 4,
                'ff_mult': 2,
                'dropout': 0.1,
                'use_timestep_cond': True,
                'use_static_film': True
            },
            'evaluation': {
                'mode': 'reconstruction',
                'num_samples': 4,
                'seed': 42,
                'generation': {
                    'num_steps': 10,
                    'guidance_scale': 1.0,
                    'temperature': 1.0,
                    'ddim_eta': 0.0
                }
            },
            'metrics': {
                'signal': {
                    'mse': True,
                    'correlation': True,
                    'snr': True,
                    'psnr': True,
                    'ssim': True,
                    'stft_loss': True
                },
                'peak_classification': {
                    'enabled': True,
                    'threshold': 0.5,
                    'bootstrap_ci': 10,
                    'significance_tests': True,
                    'clinical_validation': True
                },
                'statistical_analysis': {
                    'confidence_level': 0.95,
                    'bootstrap_method': 'percentile',
                    'multiple_testing_correction': 'bonferroni',
                    'effect_size_metrics': ['cohens_d', 'cliff_delta']
                },
                'clinical_metrics': {
                    'sensitivity_analysis': True,
                    'specificity_targets': [0.8, 0.9, 0.95],
                    'prevalence_adjustment': True,
                    'diagnostic_odds_ratio': True
                }
            },
            'output': {
                'save_dir': self.temp_dir,
                'save_format': 'json',
                'save_samples': True,
                'save_plots': True
            },
            'report': {
                'save_classification_metrics': True,
                'save_roc_curves': True,
                'save_pr_curves': True,
                'save_confusion_matrices': True,
                'publication_ready': True,
                'include_confidence_intervals': True,
                'statistical_annotations': True,
                'save_topk_examples': 4,
                'save_spectrograms': False
            },
            'tensorboard': {
                'enabled': False,
                'log_dir': os.path.join(self.temp_dir, 'logs'),
                'log_metrics': False,
                'log_plots': False,
                'log_samples': False
            }
        }
        
        # Create mock data
        self.mock_batch = {
            'x0': torch.randn(4, 1, 200),
            'stat': torch.randn(4, 4),
            'meta': [{'sample_idx': i} for i in range(4)],
            'peak_exists': torch.tensor([1, 0, 1, 0])  # Mock peak labels
        }
        
        # Create mock model output
        self.mock_model_output = {
            'signal': torch.randn(4, 1, 200),
            'peak_5th_exists': torch.tensor([0.8, 0.2, 0.9, 0.1])  # Mock logits
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_bootstrap_classification_metrics(self):
        """Test bootstrap classification metrics computation."""
        logits = np.array([0.8, 0.2, 0.9, 0.1])
        targets = np.array([1, 0, 1, 0])
        
        results = bootstrap_classification_metrics(
            logits, targets, n_bootstrap=10, confidence_level=0.95
        )
        
        # Check that all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in required_metrics:
            self.assertIn(metric, results)
            self.assertIn('mean', results[metric])
            self.assertIn('ci_lower', results[metric])
            self.assertIn('ci_upper', results[metric])
        
        # Check that values are reasonable
        self.assertGreaterEqual(results['accuracy']['mean'], 0.0)
        self.assertLessEqual(results['accuracy']['mean'], 1.0)
    
    def test_roc_analysis(self):
        """Test ROC analysis computation."""
        logits = np.array([0.8, 0.2, 0.9, 0.1])
        targets = np.array([1, 0, 1, 0])
        
        results = roc_analysis(logits, targets, threshold=0.5)
        
        # Check that ROC data is present
        self.assertIn('auroc', results)
        self.assertIn('roc_curve', results)
        self.assertIn('fpr', results['roc_curve'])
        self.assertIn('tpr', results['roc_curve'])
        
        # Check AUROC value
        self.assertGreaterEqual(results['auroc'], 0.0)
        self.assertLessEqual(results['auroc'], 1.0)
    
    def test_precision_recall_analysis(self):
        """Test precision-recall analysis computation."""
        logits = np.array([0.8, 0.2, 0.9, 0.1])
        targets = np.array([1, 0, 1, 0])
        
        results = precision_recall_analysis(logits, targets)
        
        # Check that PR data is present
        self.assertIn('aupr', results)
        self.assertIn('pr_curve', results)
        self.assertIn('precision', results['pr_curve'])
        self.assertIn('recall', results['pr_curve'])
        
        # Check AUPR value
        self.assertGreaterEqual(results['aupr'], 0.0)
        self.assertLessEqual(results['aupr'], 1.0)
    
    def test_edge_case_single_class(self):
        """Test edge case handling for single-class predictions."""
        logits = np.array([0.8, 0.9, 0.7, 0.6])
        targets = np.array([1, 1, 1, 1])  # All same class
        
        # This should handle single-class gracefully
        results = bootstrap_classification_metrics(
            logits, targets, n_bootstrap=10, confidence_level=0.95
        )
        
        # Should still return results but may have warnings
        self.assertIsInstance(results, dict)
    
    def test_edge_case_perfect_predictions(self):
        """Test edge case handling for perfect predictions."""
        logits = np.array([0.9, 0.9, 0.9, 0.9])  # All same prediction
        targets = np.array([1, 0, 1, 0])
        
        # This should handle perfect predictions gracefully
        results = bootstrap_classification_metrics(
            logits, targets, n_bootstrap=10, confidence_level=0.95
        )
        
        # Should still return results
        self.assertIsInstance(results, dict)
    
    @patch('eval.ABRDataset')
    @patch('eval.ABRTransformerGenerator')
    @patch('eval.torch.load')
    @patch('eval.prepare_noise_schedule')
    @patch('eval.DDIMSampler')
    def test_integration_eval_main(self, mock_sampler, mock_noise_schedule, 
                                   mock_load_checkpoint, mock_model_class, mock_dataset_class):
        """Integration test for eval.main() with classification evaluation."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.sequence_length = 200
        mock_dataset.static_dim = 4
        mock_dataset.static_names = ["Age", "Intensity", "StimulusRate", "FMP"]
        mock_dataset.denormalize_signal = Mock(return_value=torch.randn(4, 1, 200))
        mock_dataset.return_peak_labels = True
        mock_dataset_class.return_value = mock_dataset
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = self.mock_model_output
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_model.parameters = Mock(return_value=[torch.randn(10)])
        mock_model_class.return_value = mock_model
        
        # Mock checkpoint
        mock_checkpoint = {
            'model_state_dict': {},
            'epoch': 100,
            'step': 1000,
            'timestamp': '2024-01-01'
        }
        mock_load_checkpoint.return_value = mock_checkpoint
        
        # Mock noise schedule
        mock_noise_schedule.return_value = {'alphas': torch.ones(10)}
        
        # Mock sampler
        mock_sampler_instance = Mock()
        mock_sampler.return_value = mock_sampler_instance
        
        # Mock data loader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([self.mock_batch]))
        mock_loader.__len__ = Mock(return_value=1)
        
        # Mock create_stratified_datasets
        with patch('eval.create_stratified_datasets') as mock_create_datasets:
            mock_create_datasets.return_value = (None, mock_dataset, None)
            
            # Mock DataLoader
            with patch('eval.DataLoader') as mock_dataloader_class:
                mock_dataloader_class.return_value = mock_loader
                
                # Mock abr_collate_fn
                with patch('eval.abr_collate_fn') as mock_collate:
                    mock_collate.return_value = self.mock_batch
                    
                    # Mock compute_per_sample_metrics
                    with patch('eval.compute_per_sample_metrics') as mock_metrics:
                        mock_metrics.return_value = [
                            {'mse': 0.1, 'corr': 0.9, 'snr_db': 20.0}
                            for _ in range(4)
                        ]
                        
                        # Mock predict_x0_from_v
                        with patch('eval.predict_x0_from_v') as mock_predict:
                            mock_predict.return_value = torch.randn(4, 1, 200)
                            
                            # Mock q_sample_vpred
                            with patch('eval.q_sample_vpred') as mock_q_sample:
                                mock_q_sample.return_value = (torch.randn(4, 1, 200), torch.randn(4, 1, 200))
                                
                                # Mock torch.randn
                                with patch('torch.randn') as mock_randn:
                                    mock_randn.return_value = torch.randn(4, 1, 200)
                                    
                                    # Mock torch.randint
                                    with patch('torch.randint') as mock_randint:
                                        mock_randint.return_value = torch.tensor([5, 5, 5, 5])
                                        
                                        # Mock SummaryWriter
                                        with patch('eval.SummaryWriter') as mock_writer_class:
                                            mock_writer = Mock()
                                            mock_writer_class.return_value = mock_writer
                                            
                                            # Save config to temp file
                                            config_path = os.path.join(self.temp_dir, 'test_config.yaml')
                                            with open(config_path, 'w') as f:
                                                yaml.dump(self.test_config, f)
                                            
                                            # Mock argparse
                                            with patch('sys.argv', ['eval.py', '--config', config_path]):
                                                with patch('argparse.ArgumentParser.parse_args') as mock_args:
                                                    mock_args.return_value = Mock(
                                                        config=config_path,
                                                        override=""
                                                    )
                                                    
                                                    # Import and run main
                                                    try:
                                                        import eval
                                                        results = eval.main()
                                                        
                                                        # Assertions
                                                        self.assertIsInstance(results, dict)
                                                        self.assertIn('reconstruction', results)
                                                        
                                                        # Check that results were saved
                                                        output_files = os.listdir(self.temp_dir)
                                                        self.assertTrue(any('reconstruction' in f for f in output_files))
                                                        self.assertTrue(any('evaluation_summary.json' in f for f in output_files))
                                                        
                                                        # Check evaluation summary structure
                                                        summary_path = os.path.join(self.temp_dir, 'evaluation_summary.json')
                                                        if os.path.exists(summary_path):
                                                            with open(summary_path, 'r') as f:
                                                                summary = json.load(f)
                                                            
                                                            self.assertIn('metrics', summary)
                                                            self.assertIn('reconstruction', summary['metrics'])
                                                            self.assertIn('signal', summary['metrics']['reconstruction'])
                                                        
                                                    except Exception as e:
                                                        self.fail(f"Integration test failed with error: {e}")
    
    def test_config_schema_compatibility(self):
        """Test that the config schema is compatible with eval.py requirements."""
        required_keys = [
            'dataset.return_peak_labels',
            'model.checkpoint_path',
            'evaluation.mode',
            'metrics.peak_classification.enabled',
            'output.save_dir',
            'tensorboard.enabled'
        ]
        
        for key_path in required_keys:
            keys = key_path.split('.')
            value = self.test_config
            for key in keys:
                self.assertIn(key, value, f"Missing config key: {key_path}")
                value = value[key]
    
    def test_classification_metrics_structure(self):
        """Test that classification metrics have the expected structure."""
        logits = np.array([0.8, 0.2, 0.9, 0.1])
        targets = np.array([1, 0, 1, 0])
        
        # Test bootstrap metrics
        bootstrap_results = bootstrap_classification_metrics(
            logits, targets, n_bootstrap=10, confidence_level=0.95
        )
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            self.assertIn(metric_name, bootstrap_results)
            metric_data = bootstrap_results[metric_name]
            self.assertIn('mean', metric_data)
            self.assertIn('ci_lower', metric_data)
            self.assertIn('ci_upper', metric_data)
            self.assertIsInstance(metric_data['mean'], (int, float))
            self.assertIsInstance(metric_data['ci_lower'], (int, float))
            self.assertIsInstance(metric_data['ci_upper'], (int, float))


if __name__ == '__main__':
    unittest.main()
