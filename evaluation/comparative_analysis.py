"""
Comparative Analysis Module for ABR Transformer Evaluation

This module provides comprehensive tools for comparing multiple evaluation runs,
performing ablation study analysis, and generating publication-ready comparison reports.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from datetime import datetime

from .analysis import comparative_statistical_analysis, statistical_significance_tests, perform_mcnemar_test
from .visualization import (plot_classification_metrics_comparison, 
                           plot_ablation_study_results, create_publication_figure)


class ComparativeAnalyzer:
    """
    Comprehensive comparative analysis for multiple evaluation runs.
    
    This class provides functionality to:
    - Load and compare multiple evaluation results
    - Perform statistical significance testing
    - Generate ablation study analysis
    - Create publication-ready comparison reports
    """
    
    def __init__(self, evaluation_dirs: List[str], baseline_run: Optional[str] = None):
        """
        Initialize the comparative analyzer.
        
        Args:
            evaluation_dirs: List of paths to evaluation result directories
            baseline_run: Path to baseline run for comparison (optional)
        """
        self.evaluation_dirs = evaluation_dirs
        self.baseline_run = baseline_run
        self.results = {}
        self.metadata = {}
        self.comparison_results = {}
        
        # Load evaluation results
        self._load_all_results()
        
        # Validate comparability
        self._validate_comparability()
    
    def _load_all_results(self) -> None:
        """Load evaluation results from all directories."""
        for eval_dir in self.evaluation_dirs:
            try:
                result_data = self.load_evaluation_results(eval_dir)
                if result_data:
                    run_name = Path(eval_dir).name
                    self.results[run_name] = result_data['metrics']
                    self.metadata[run_name] = result_data['metadata']
                    print(f"Loaded results from {run_name}")
            except Exception as e:
                print(f"Warning: Failed to load results from {eval_dir}: {e}")
    
    def _validate_comparability(self) -> None:
        """Validate that all runs can be compared."""
        if len(self.results) < 2:
            warnings.warn("Need at least 2 evaluation runs for comparison")
            return
        
        # Check if all runs have the same metrics
        first_run = list(self.results.keys())[0]
        first_metrics = set(self.results[first_run].keys())
        
        for run_name, metrics in self.results.items():
            current_metrics = set(metrics.keys())
            if current_metrics != first_metrics:
                warnings.warn(f"Run {run_name} has different metrics than {first_run}")
    
    @staticmethod
    def load_evaluation_results(evaluation_dir: str) -> Optional[Dict]:
        """
        Load evaluation results from a directory.
        
        Args:
            evaluation_dir: Path to evaluation result directory
            
        Returns:
            Dictionary with metrics and metadata, or None if loading fails
        """
        eval_path = Path(evaluation_dir)
        
        # Look for evaluation summary file
        summary_files = list(eval_path.glob("*.json")) + list(eval_path.glob("evaluation_summary.json"))
        
        if not summary_files:
            print(f"No evaluation summary found in {evaluation_dir}")
            return None
        
        summary_file = summary_files[0]
        
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Extract metrics and metadata
            metrics = data.get('metrics', {})
            metadata = {
                'evaluation_dir': evaluation_dir,
                'timestamp': data.get('timestamp', ''),
                'model_config': data.get('model_config', {}),
                'dataset_info': data.get('dataset_info', {}),
                'evaluation_config': data.get('evaluation_config', {})
            }
            
            return {
                'metrics': metrics,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error loading results from {summary_file}: {e}")
            return None
    
    def compare_signal_generation_metrics(self, 
                                        metrics_to_compare: List[str] = None) -> Dict:
        """
        Compare signal generation metrics across runs.
        
        Args:
            metrics_to_compare: List of metrics to compare (default: all available)
            
        Returns:
            Dictionary with comparison results and statistical tests
        """
        if not metrics_to_compare:
            # Default signal generation metrics
            metrics_to_compare = ['mse', 'correlation', 'snr', 'psnr', 'ssim', 'l1', 'stft_l1']
        
        comparison_results = {}
        skipped_metrics = []
        
        # Get baseline if specified
        baseline_metrics = None
        if self.baseline_run and self.baseline_run in self.results:
            baseline_metrics = self.results[self.baseline_run]
        
        # Compare each run against baseline or other runs
        for run_name, metrics in self.results.items():
            if run_name == self.baseline_run:
                continue
                
            run_comparison = {}
            
            for metric in metrics_to_compare:
                if baseline_metrics and metric in metrics and metric in baseline_metrics:
                    # Calculate difference
                    diff = metrics[metric] - baseline_metrics[metric]
                    
                    # Perform statistical test if we have multiple samples
                    # For now, we'll just store the difference
                    run_comparison[metric] = {
                        'baseline_value': baseline_metrics[metric],
                        'current_value': metrics[metric],
                        'difference': diff,
                        'relative_improvement': (diff / baseline_metrics[metric]) * 100 if baseline_metrics[metric] != 0 else 0
                    }
                elif metric not in metrics:
                    if metric not in skipped_metrics:
                        skipped_metrics.append(metric)
                elif baseline_metrics and metric not in baseline_metrics:
                    if f"{metric} (baseline)" not in skipped_metrics:
                        skipped_metrics.append(f"{metric} (baseline)")
            
            comparison_results[run_name] = run_comparison
        
        # Log skipped metrics
        if skipped_metrics:
            print(f"Warning: Skipped signal generation metrics not found in all runs: {', '.join(skipped_metrics)}")
        
        self.comparison_results['signal_generation'] = comparison_results
        return comparison_results
    
    def compare_classification_metrics(self, 
                                     metrics_to_compare: List[str] = None) -> Dict:
        """
        Compare peak classification metrics across runs.
        
        Args:
            metrics_to_compare: List of metrics to compare (default: all available)
            
        Returns:
            Dictionary with comparison results and statistical tests
        """
        if not metrics_to_compare:
            # Default classification metrics
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
        
        comparison_results = {}
        skipped_metrics = []
        
        # Get baseline if specified
        baseline_metrics = None
        if self.baseline_run and self.baseline_run in self.results:
            baseline_metrics = self.results[self.baseline_run]
        
        # Compare each run against baseline or other runs
        for run_name, metrics in self.results.items():
            if run_name == self.baseline_run:
                continue
                
            run_comparison = {}
            
            for metric in metrics_to_compare:
                if baseline_metrics and metric in metrics and metric in baseline_metrics:
                    # Calculate difference
                    diff = metrics[metric] - baseline_metrics[metric]
                    
                    # Perform statistical test if we have multiple samples
                    # For now, we'll just store the difference
                    run_comparison[metric] = {
                        'baseline_value': baseline_metrics[metric],
                        'current_value': metrics[metric],
                        'difference': diff,
                        'relative_improvement': (diff / baseline_metrics[metric]) * 100 if baseline_metrics[metric] != 0 else 0
                    }
                elif metric not in metrics:
                    if metric not in skipped_metrics:
                        skipped_metrics.append(metric)
                elif baseline_metrics and metric not in baseline_metrics:
                    if f"{metric} (baseline)" not in skipped_metrics:
                        skipped_metrics.append(f"{metric} (baseline)")
            
            comparison_results[run_name] = run_comparison
        
        # Log skipped metrics
        if skipped_metrics:
            print(f"Warning: Skipped classification metrics not found in all runs: {', '.join(skipped_metrics)}")
        
        self.comparison_results['classification'] = comparison_results
        return comparison_results
    
    def ablation_study_analysis(self) -> Dict:
        """
        Analyze the effect of individual architectural components.
        
        Returns:
            Dictionary with ablation study results
        """
        if not self.baseline_run or self.baseline_run not in self.results:
            print("Warning: No baseline run specified for ablation study")
            return {}
        
        baseline_metrics = self.results[self.baseline_run]
        ablation_results = []
        
        for run_name, metrics in self.results.items():
            if run_name == self.baseline_run:
                continue
            
            # Extract component name from run name (assuming naming convention)
            component_name = self._extract_component_name(run_name)
            
            # Calculate effect sizes for each metric
            effect_sizes = {}
            for metric_name, metric_value in metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    # Calculate Cohen's d effect size
                    effect_size = self._calculate_effect_size(baseline_value, metric_value)
                    effect_sizes[metric_name] = effect_size
            
            ablation_result = {
                'component_name': component_name,
                'run_name': run_name,
                'effect_sizes': effect_sizes,
                'metrics': metrics,
                'baseline_metrics': baseline_metrics
            }
            
            ablation_results.append(ablation_result)
        
        # Rank components by their contribution
        ablation_results = self._rank_ablation_components(ablation_results)
        
        self.comparison_results['ablation_study'] = ablation_results
        return ablation_results
    
    def _extract_component_name(self, run_name: str) -> str:
        """Extract component name from run name."""
        # Common patterns for ablation study naming
        if 'no_' in run_name.lower():
            return run_name.replace('no_', '').replace('_', ' ').title()
        elif 'ablation_' in run_name.lower():
            return run_name.replace('ablation_', '').replace('_', ' ').title()
        else:
            return run_name.replace('_', ' ').title()
    
    def _calculate_effect_size(self, baseline: float, current: float, 
                              baseline_std: float = None, current_std: float = None) -> float:
        """
        Calculate Cohen's d effect size with real or estimated standard deviations.
        
        Args:
            baseline: Baseline metric value
            current: Current metric value
            baseline_std: Standard deviation of baseline (if available)
            current_std: Standard deviation of current (if available)
        
        Returns:
            Cohen's d effect size
        """
        diff = current - baseline
        
        # Use real sample-level stats if available, otherwise report uncertainty
        if baseline_std is not None and current_std is not None:
            pooled_std = np.sqrt((baseline_std**2 + current_std**2) / 2)
            if pooled_std > 0:
                return diff / pooled_std
            else:
                print("Warning: Zero pooled standard deviation, using simplified effect size")
                return diff / 0.2  # Fallback
        else:
            # Report uncertainty when sample-level stats are not available
            print(f"Warning: Sample-level statistics not available for effect size calculation. Using simplified estimate.")
            # Use heuristic based on metric type
            if abs(baseline) > 0:
                # Use relative change as proxy for effect size
                relative_change = abs(diff / baseline)
                if relative_change < 0.05:
                    return diff / 0.2  # Small effect
                elif relative_change < 0.20:
                    return diff / 0.5  # Medium effect
                else:
                    return diff / 0.8  # Large effect
            else:
                return diff / 0.2  # Default small effect threshold
    
    def _rank_ablation_components(self, ablation_results: List[Dict]) -> List[Dict]:
        """Rank ablation components by their overall effect."""
        for result in ablation_results:
            # Calculate average effect size across metrics
            effect_sizes = list(result['effect_sizes'].values())
            if effect_sizes:
                result['average_effect_size'] = np.mean(effect_sizes)
                result['max_effect_size'] = np.max(effect_sizes)
            else:
                result['average_effect_size'] = 0
                result['max_effect_size'] = 0
        
        # Sort by average effect size (descending)
        ablation_results.sort(key=lambda x: abs(x['average_effect_size']), reverse=True)
        
        return ablation_results
    
    def generate_comparison_table(self, output_format: str = 'csv') -> str:
        """
        Generate publication-ready comparison table.
        
        Args:
            output_format: Output format ('csv', 'latex', 'html')
            
        Returns:
            Path to generated comparison table
        """
        if not self.comparison_results:
            print("No comparison results available. Run comparison methods first.")
            return ""
        
        # Prepare data for table
        table_data = []
        
        for run_name in self.results.keys():
            if run_name == self.baseline_run:
                continue
                
            row = {'Run': run_name}
            
            # Add signal generation metrics
            if 'signal_generation' in self.comparison_results and run_name in self.comparison_results['signal_generation']:
                for metric, data in self.comparison_results['signal_generation'][run_name].items():
                    row[f'{metric}_baseline'] = f"{data['baseline_value']:.4f}"
                    row[f'{metric}_current'] = f"{data['current_value']:.4f}"
                    row[f'{metric}_improvement'] = f"{data['relative_improvement']:+.2f}%"
            
            # Add classification metrics
            if 'classification' in self.comparison_results and run_name in self.comparison_results['classification']:
                for metric, data in self.comparison_results['classification'][run_name].items():
                    row[f'{metric}_baseline'] = f"{data['baseline_value']:.4f}"
                    row[f'{metric}_current'] = f"{data['current_value']:.4f}"
                    row[f'{metric}_improvement'] = f"{data['relative_improvement']:+.2f}%"
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Generate output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("comparative_analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        if output_format == 'csv':
            output_path = output_dir / f"comparison_table_{timestamp}.csv"
            df.to_csv(output_path, index=False)
        elif output_format == 'latex':
            output_path = output_dir / f"comparison_table_{timestamp}.tex"
            latex_table = df.to_latex(index=False, float_format="%.4f")
            with open(output_path, 'w') as f:
                f.write(latex_table)
        elif output_format == 'html':
            output_path = output_dir / f"comparison_table_{timestamp}.html"
            html_table = df.to_html(index=False, float_format="%.4f")
            with open(output_path, 'w') as f:
                f.write(html_table)
        
        print(f"Comparison table saved to {output_path}")
        return str(output_path)
    
    def generate_comparative_report(self, output_dir: str = None) -> str:
        """
        Create comprehensive HTML report with all comparative analyses.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to generated report
        """
        if output_dir is None:
            output_dir = "comparative_analysis_results"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_path / f"comparative_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Comparative report saved to {report_path}")
        return str(report_path)
    
    def _generate_html_report(self) -> str:
        """Generate HTML content for the comparative report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ABR Transformer Comparative Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .improvement {{ color: green; font-weight: bold; }}
                .degradation {{ color: red; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ABR Transformer Comparative Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Baseline Run: {self.baseline_run or 'None specified'}</p>
            </div>
        """
        
        # Add run summary
        html += f"""
            <div class="section">
                <h2>Evaluation Runs Summary</h2>
                <p>Total runs analyzed: {len(self.results)}</p>
                <ul>
        """
        
        for run_name, metadata in self.metadata.items():
            html += f"<li><strong>{run_name}</strong>: {metadata.get('timestamp', 'N/A')}</li>"
        
        html += "</ul></div>"
        
        # Add signal generation comparison
        if 'signal_generation' in self.comparison_results:
            html += self._generate_comparison_section_html('Signal Generation', 'signal_generation')
        
        # Add classification comparison
        if 'classification' in self.comparison_results:
            html += self._generate_comparison_section_html('Classification', 'classification')
        
        # Add ablation study results
        if 'ablation_study' in self.comparison_results:
            html += self._generate_ablation_section_html()
        
        html += """
            </body>
        </html>
        """
        
        return html
    
    def _generate_comparison_section_html(self, title: str, comparison_type: str) -> str:
        """Generate HTML for a comparison section."""
        html = f"""
            <div class="section">
                <h2>{title} Metrics Comparison</h2>
                <table>
                    <tr>
                        <th>Run</th>
                        <th>Metric</th>
                        <th>Baseline</th>
                        <th>Current</th>
                        <th>Improvement</th>
                    </tr>
        """
        
        for run_name, metrics in self.comparison_results[comparison_type].items():
            for metric, data in metrics.items():
                improvement_class = "improvement" if data['relative_improvement'] > 0 else "degradation"
                html += f"""
                    <tr>
                        <td>{run_name}</td>
                        <td>{metric.upper()}</td>
                        <td>{data['baseline_value']:.4f}</td>
                        <td>{data['current_value']:.4f}</td>
                        <td class="{improvement_class}">{data['relative_improvement']:+.2f}%</td>
                    </tr>
                """
        
        html += "</table></div>"
        return html
    
    def _generate_ablation_section_html(self) -> str:
        """Generate HTML for ablation study section."""
        html = """
            <div class="section">
                <h2>Ablation Study Results</h2>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Average Effect Size</th>
                        <th>Max Effect Size</th>
                        <th>Impact</th>
                    </tr>
        """
        
        for result in self.comparison_results['ablation_study']:
            effect_size = abs(result['average_effect_size'])
            if effect_size < 0.2:
                impact = "Negligible"
            elif effect_size < 0.5:
                impact = "Small"
            elif effect_size < 0.8:
                impact = "Medium"
            else:
                impact = "Large"
            
            html += f"""
                <tr>
                    <td>{result['component_name']}</td>
                    <td>{result['average_effect_size']:.3f}</td>
                    <td>{result['max_effect_size']:.3f}</td>
                    <td>{impact}</td>
                </tr>
            """
        
        html += "</table></div>"
        return html
    
    def cross_validation_analysis(self) -> Dict:
        """
        Analyze results across multiple cross-validation folds.
        
        Returns:
            Dictionary with cross-validation stability analysis
        """
        # This would require multiple evaluation runs with different seeds
        # For now, return empty results
        return {
            'message': 'Cross-validation analysis requires multiple runs with different seeds',
            'stability_metrics': {}
        }
    
    def get_performance_rankings(self, metrics: List[str] = None) -> Dict:
        """
        Get performance rankings across all runs for specified metrics.
        
        Args:
            metrics: List of metrics to rank (default: all available)
            
        Returns:
            Dictionary with rankings for each metric
        """
        if not metrics:
            # Get all available metrics from first run
            first_run = list(self.results.keys())[0]
            metrics = list(self.results[first_run].keys())
        
        rankings = {}
        
        for metric in metrics:
            # Collect values for this metric across all runs
            metric_values = []
            run_names = []
            
            for run_name, run_metrics in self.results.items():
                if metric in run_metrics:
                    metric_values.append(run_metrics[metric])
                    run_names.append(run_name)
            
            if metric_values:
                # Sort by performance (higher is better for most metrics)
                # For MSE, lower is better
                reverse_sort = metric not in ['mse', 'mae']  # These are error metrics
                
                sorted_indices = np.argsort(metric_values)
                if reverse_sort:
                    sorted_indices = sorted_indices[::-1]
                
                # Create ranking
                ranking = []
                for i, idx in enumerate(sorted_indices):
                    ranking.append({
                        'rank': i + 1,
                        'run_name': run_names[idx],
                        'value': metric_values[idx]
                    })
                
                rankings[metric] = ranking
        
        return rankings
    
    def generate_statistical_summary(self) -> Dict:
        """
        Generate comprehensive statistical summary of all comparisons.
        
        Returns:
            Dictionary with statistical summary
        """
        summary = {
            'total_runs': len(self.results),
            'baseline_run': self.baseline_run,
            'comparison_summary': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Summary of signal generation comparisons
        if 'signal_generation' in self.comparison_results:
            signal_summary = self._summarize_comparisons('signal_generation')
            summary['comparison_summary']['signal_generation'] = signal_summary
        
        # Summary of classification comparisons
        if 'classification' in self.comparison_results:
            class_summary = self._summarize_comparisons('classification')
            summary['comparison_summary']['classification'] = class_summary
        
        # Ablation study summary
        if 'ablation_study' in self.comparison_results:
            ablation_summary = self._summarize_ablation_study()
            summary['comparison_summary']['ablation_study'] = ablation_summary
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _summarize_comparisons(self, comparison_type: str) -> Dict:
        """Summarize comparison results for a specific type."""
        comparisons = self.comparison_results[comparison_type]
        
        summary = {
            'total_comparisons': len(comparisons),
            'improvements': 0,
            'degradations': 0,
            'best_improvement': 0,
            'worst_degradation': 0
        }
        
        for run_name, metrics in comparisons.items():
            for metric, data in metrics.items():
                if data['relative_improvement'] > 0:
                    summary['improvements'] += 1
                    summary['best_improvement'] = max(summary['best_improvement'], data['relative_improvement'])
                else:
                    summary['degradations'] += 1
                    summary['worst_degradation'] = min(summary['worst_degradation'], data['relative_improvement'])
        
        return summary
    
    def _summarize_ablation_study(self) -> Dict:
        """Summarize ablation study results."""
        ablation_results = self.comparison_results['ablation_study']
        
        summary = {
            'total_components': len(ablation_results),
            'high_impact_components': 0,
            'medium_impact_components': 0,
            'low_impact_components': 0,
            'top_components': []
        }
        
        for result in ablation_results:
            effect_size = abs(result['average_effect_size'])
            if effect_size >= 0.8:
                summary['high_impact_components'] += 1
            elif effect_size >= 0.5:
                summary['medium_impact_components'] += 1
            else:
                summary['low_impact_components'] += 1
        
        # Get top 3 components
        top_components = sorted(ablation_results, 
                              key=lambda x: abs(x['average_effect_size']), 
                              reverse=True)[:3]
        
        summary['top_components'] = [
            {
                'name': comp['component_name'],
                'effect_size': comp['average_effect_size'],
                'run_name': comp['run_name']
            }
            for comp in top_components
        ]
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Signal generation recommendations
        if 'signal_generation' in self.comparison_results:
            signal_summary = self.comparison_results['signal_generation']
            improvements = sum(1 for run in signal_summary.values() 
                            for metric in run.values() 
                            if metric['relative_improvement'] > 0)
            
            if improvements > 0:
                recommendations.append("Consider architectural changes that improve signal generation metrics")
        
        # Classification recommendations
        if 'classification' in self.comparison_results:
            class_summary = self.comparison_results['classification']
            improvements = sum(1 for run in class_summary.values() 
                            for metric in run.values() 
                            if metric['relative_improvement'] > 0)
            
            if improvements > 0:
                recommendations.append("Focus on components that enhance peak classification performance")
        
        # Ablation study recommendations
        if 'ablation_study' in self.comparison_results:
            ablation_summary = self.comparison_results['ablation_study']
            high_impact = ablation_summary['high_impact_components']
            
            if high_impact > 0:
                recommendations.append(f"Prioritize {high_impact} high-impact architectural components")
        
        if not recommendations:
            recommendations.append("No specific recommendations based on current analysis")
        
        return recommendations
    
    def enhanced_classification_comparison(self, 
                                         metrics_to_compare: List[str] = None) -> Dict:
        """
        Enhanced comparison with statistical significance tests and real sample-level stats.
        
        Args:
            metrics_to_compare: List of metrics to compare (default: all available)
            
        Returns:
            Dictionary with enhanced comparison results including statistical tests
        """
        if not metrics_to_compare:
            # Default classification metrics
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
        
        comparison_results = {}
        skipped_metrics = []
        
        # Get baseline if specified
        baseline_metrics = None
        if self.baseline_run and self.baseline_run in self.results:
            baseline_metrics = self.results[self.baseline_run]
        
        # Compare each run against baseline or other runs
        for run_name, metrics in self.results.items():
            if run_name == self.baseline_run:
                continue
                
            run_comparison = {}
            
            for metric in metrics_to_compare:
                if baseline_metrics and metric in metrics and metric in baseline_metrics:
                    # Calculate difference
                    diff = metrics[metric] - baseline_metrics[metric]
                    
                    # Get standard deviations if available (from confidence intervals)
                    baseline_std = None
                    current_std = None
                    if isinstance(baseline_metrics[metric], dict) and 'std' in baseline_metrics[metric]:
                        baseline_std = baseline_metrics[metric]['std']
                    if isinstance(metrics[metric], dict) and 'std' in metrics[metric]:
                        current_std = metrics[metric]['std']
                    
                    # Calculate effect size with real or estimated standard deviations
                    effect_size = self._calculate_effect_size(
                        baseline_metrics[metric], metrics[metric], 
                        baseline_std, current_std
                    )
                    
                    # Perform statistical significance tests if sample-level data is available
                    statistical_tests = {}
                    if 'sample_data' in self.metadata.get(run_name, {}) and 'sample_data' in self.metadata.get(self.baseline_run, {}):
                        try:
                            # Get sample-level predictions and targets if available
                            current_sample_data = self.metadata[run_name]['sample_data']
                            baseline_sample_data = self.metadata[self.baseline_run]['sample_data']
                            
                            if 'predictions' in current_sample_data and 'targets' in current_sample_data:
                                # Perform McNemar's test for paired classification
                                mcnemar_result = perform_mcnemar_test(
                                    baseline_sample_data, current_sample_data
                                )
                                statistical_tests['mcnemar_test'] = mcnemar_result
                                
                                # Perform statistical significance tests
                                if 'logits' in current_sample_data:
                                    significance_results = statistical_significance_tests(
                                        current_sample_data['logits'], 
                                        current_sample_data['targets'],
                                        prevalence=current_sample_data.get('prevalence'),
                                        correction='bonferroni'
                                    )
                                    statistical_tests['significance_tests'] = significance_results
                        except Exception as e:
                            print(f"Warning: Could not perform statistical tests for {metric}: {e}")
                    
                    run_comparison[metric] = {
                        'baseline_value': baseline_metrics[metric],
                        'current_value': metrics[metric],
                        'difference': diff,
                        'relative_improvement': (diff / baseline_metrics[metric]) * 100 if baseline_metrics[metric] != 0 else 0,
                        'effect_size': effect_size,
                        'statistical_tests': statistical_tests
                    }
                elif metric not in metrics:
                    if metric not in skipped_metrics:
                        skipped_metrics.append(metric)
                elif baseline_metrics and metric not in baseline_metrics:
                    if f"{metric} (baseline)" not in skipped_metrics:
                        skipped_metrics.append(f"{metric} (baseline)")
            
            comparison_results[run_name] = run_comparison
        
        # Log skipped metrics
        if skipped_metrics:
            print(f"Warning: Skipped classification metrics not found in all runs: {', '.join(skipped_metrics)}")
        
        self.comparison_results['enhanced_classification'] = comparison_results
        return comparison_results


def validate_comparability(run1: Dict, run2: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that two evaluation runs can be compared.
    
    Args:
        run1: First evaluation run data
        run2: Second evaluation run data
        
    Returns:
        Tuple of (is_comparable, list_of_issues)
    """
    issues = []
    
    # Check if runs have the same metrics
    metrics1 = set(run1.keys())
    metrics2 = set(run2.keys())
    
    if metrics1 != metrics2:
        missing_in_1 = metrics2 - metrics1
        missing_in_2 = metrics1 - metrics2
        if missing_in_1:
            issues.append(f"Run 2 has metrics not in Run 1: {missing_in_1}")
        if missing_in_2:
            issues.append(f"Run 1 has metrics not in Run 2: {missing_in_2}")
    
    # Check if runs have similar value ranges
    common_metrics = metrics1.intersection(metrics2)
    for metric in common_metrics:
        val1 = run1[metric]
        val2 = run2[metric]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 != 0 and abs((val2 - val1) / val1) > 10:  # 10x difference threshold
                issues.append(f"Large difference in {metric}: {val1} vs {val2}")
    
    is_comparable = len(issues) == 0
    return is_comparable, issues


def extract_configuration_differences(config1: Dict, config2: Dict) -> Dict:
    """
    Extract differences between two configuration dictionaries.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary with configuration differences
    """
    differences = {}
    
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in all_keys:
        if key not in config1:
            differences[key] = {'type': 'added', 'value': config2[key]}
        elif key not in config2:
            differences[key] = {'type': 'removed', 'value': config1[key]}
        elif config1[key] != config2[key]:
            differences[key] = {
                'type': 'modified',
                'old_value': config1[key],
                'new_value': config2[key]
            }
    
    return differences


def calculate_performance_rankings(metrics_data: List[Dict], 
                                 metric_names: List[str]) -> Dict:
    """
    Calculate performance rankings across multiple models.
    
    Args:
        metrics_data: List of metrics dictionaries for each model
        metric_names: List of metric names to rank
        
    Returns:
        Dictionary with rankings for each metric
    """
    rankings = {}
    
    for metric in metric_names:
        # Collect values for this metric
        values = []
        model_indices = []
        
        for i, model_data in enumerate(metrics_data):
            if metric in model_data:
                values.append(model_data[metric])
                model_indices.append(i)
        
        if values:
            # Sort by performance (higher is better for most metrics)
            reverse_sort = metric not in ['mse', 'mae']  # Error metrics
            
            sorted_indices = np.argsort(values)
            if reverse_sort:
                sorted_indices = sorted_indices[::-1]
            
            # Create ranking
            ranking = []
            for rank, idx in enumerate(sorted_indices):
                ranking.append({
                    'rank': rank + 1,
                    'model_index': model_indices[idx],
                    'value': values[idx]
                })
            
            rankings[metric] = ranking
    
    return rankings


def generate_statistical_summary(comparison_results: Dict) -> str:
    """
    Generate a text summary of statistical analysis results.
    
    Args:
        comparison_results: Results from statistical analysis
        
    Returns:
        Formatted text summary
    """
    summary_lines = ["Statistical Analysis Summary", "=" * 30, ""]
    
    for metric, results in comparison_results.items():
        if isinstance(results, dict) and 'p_value' in results:
            summary_lines.append(f"{metric.upper()}:")
            summary_lines.append(f"  P-value: {results['p_value']:.4f}")
            summary_lines.append(f"  Significant: {'Yes' if results['significant'] else 'No'}")
            
            if 'effect_size' in results:
                summary_lines.append(f"  Effect size: {results['effect_size']:.3f}")
                summary_lines.append(f"  Interpretation: {results.get('interpretation', 'N/A')}")
            
            summary_lines.append("")
    
    return "\n".join(summary_lines)
