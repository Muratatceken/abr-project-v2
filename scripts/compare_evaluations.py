#!/usr/bin/env python3
"""
ABR Transformer Evaluation Comparison Script

This script provides a command-line interface for comparing multiple evaluation runs,
performing ablation study analysis, and generating publication-ready comparison reports.

Usage:
    python compare_evaluations.py --input-dirs results/run1 results/run2 --output-dir comparison_results
    python compare_evaluations.py --input-dirs "results/ablation_*" --baseline results/baseline
    python compare_evaluations.py --input-dirs results/* --format html --significance-level 0.01
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import glob

# Add parent directory to path to import evaluation modules
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.comparative_analysis import ComparativeAnalyzer


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multiple ABR Transformer evaluation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two specific runs
    python compare_evaluations.py --input-dirs results/run1 results/run2
    
    # Compare all ablation study runs with baseline
    python compare_evaluations.py --input-dirs "results/ablation_*" --baseline results/baseline
    
    # Generate HTML report with specific metrics
    python compare_evaluations.py --input-dirs results/* --metrics accuracy f1 auroc --format html
    
    # Batch process all evaluation directories
    python compare_evaluations.py --input-dirs "results/*" --output-dir comparison_results
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input-dirs',
        nargs='+',
        required=True,
        help='List of evaluation result directories to compare'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='comparative_analysis_results',
        help='Directory to save comparison results (default: comparative_analysis_results)'
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        help='Specify baseline run for comparison'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['accuracy', 'f1', 'auroc', 'mse', 'correlation'],
        help='Select specific metrics to compare (default: accuracy, f1, auroc, mse, correlation)'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'html', 'pdf', 'latex'],
        default='html',
        help='Output format for comparison tables (default: html)'
    )
    
    parser.add_argument(
        '--significance-level',
        type=float,
        default=0.05,
        help='Statistical significance threshold (default: 0.05)'
    )
    
    parser.add_argument(
        '--effect-size-threshold',
        type=float,
        default=0.2,
        help='Minimum effect size for practical significance (default: 0.2)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output for detailed statistical information'
    )
    
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Enable batch processing mode for multiple directories'
    )
    
    parser.add_argument(
        '--group-by',
        type=str,
        choices=['experiment', 'architecture', 'config'],
        help='Group comparisons by experiment type, architecture, or configuration'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save comparison plots and visualizations'
    )
    
    parser.add_argument(
        '--publication-ready',
        action='store_true',
        help='Generate publication-ready figures with high resolution'
    )
    
    return parser.parse_args()


def expand_glob_patterns(input_dirs: List[str]) -> List[str]:
    """Expand glob patterns in input directories."""
    expanded_dirs = []
    
    for pattern in input_dirs:
        if '*' in pattern or '?' in pattern:
            # This is a glob pattern
            matched_dirs = glob.glob(pattern)
            if matched_dirs:
                expanded_dirs.extend(matched_dirs)
            else:
                print(f"Warning: No directories matched pattern '{pattern}'")
        else:
            # This is a literal path
            expanded_dirs.append(pattern)
    
    return expanded_dirs


def validate_input_directories(input_dirs: List[str]) -> List[str]:
    """Validate that input directories exist and contain evaluation results."""
    valid_dirs = []
    
    for dir_path in input_dirs:
        path = Path(dir_path)
        
        if not path.exists():
            print(f"Warning: Directory does not exist: {dir_path}")
            continue
        
        if not path.is_dir():
            print(f"Warning: Path is not a directory: {dir_path}")
            continue
        
        # Check if directory contains evaluation results
        evaluation_files = list(path.glob("*.json")) + list(path.glob("evaluation_summary.json"))
        
        if not evaluation_files:
            print(f"Warning: No evaluation results found in {dir_path}")
            continue
        
        valid_dirs.append(dir_path)
    
    if not valid_dirs:
        print("Error: No valid evaluation directories found")
        sys.exit(1)
    
    print(f"Found {len(valid_dirs)} valid evaluation directories:")
    for dir_path in valid_dirs:
        print(f"  - {dir_path}")
    
    return valid_dirs


def group_evaluation_runs(input_dirs: List[str], group_by: str) -> dict:
    """Group evaluation runs by specified criteria."""
    grouped_runs = {}
    
    for dir_path in input_dirs:
        path = Path(dir_path)
        
        if group_by == 'experiment':
            # Group by experiment type (e.g., ablation, hyperparameter, etc.)
            if 'ablation' in path.name.lower():
                group_key = 'ablation_study'
            elif 'hyperparam' in path.name.lower() or 'hp' in path.name.lower():
                group_key = 'hyperparameter_tuning'
            elif 'baseline' in path.name.lower():
                group_key = 'baseline'
            else:
                group_key = 'other'
        
        elif group_by == 'architecture':
            # Group by architecture type
            if 'enhanced' in path.name.lower():
                group_key = 'enhanced'
            elif 'baseline' in path.name.lower():
                group_key = 'baseline'
            elif 'ablated' in path.name.lower():
                group_key = 'ablated'
            else:
                group_key = 'unknown'
        
        elif group_by == 'config':
            # Group by configuration file
            config_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
            if config_files:
                config_name = config_files[0].stem
                group_key = config_name
            else:
                group_key = 'no_config'
        
        else:
            group_key = 'ungrouped'
        
        if group_key not in grouped_runs:
            grouped_runs[group_key] = []
        
        grouped_runs[group_key].append(dir_path)
    
    return grouped_runs


def run_comparative_analysis(input_dirs: List[str], baseline: Optional[str], 
                           metrics: List[str], output_dir: str, args) -> None:
    """Run the main comparative analysis."""
    print(f"\nStarting comparative analysis...")
    print(f"Input directories: {len(input_dirs)}")
    print(f"Baseline: {baseline or 'None'}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Output directory: {output_dir}")
    
    # Initialize comparative analyzer
    analyzer = ComparativeAnalyzer(input_dirs, baseline_run=baseline)
    
    if not analyzer.results:
        print("Error: No evaluation results could be loaded")
        return
    
    print(f"Successfully loaded {len(analyzer.results)} evaluation runs")
    
    # Separate signal and classification metrics
    signal_metrics = ['mse', 'correlation', 'snr', 'psnr', 'ssim', 'l1', 'stft_l1']
    classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    
    # Filter requested metrics into appropriate categories
    requested_signal_metrics = [m for m in metrics if m in signal_metrics]
    requested_classification_metrics = [m for m in metrics if m in classification_metrics]
    
    # Run comparisons with separate lists
    print("\nRunning signal generation comparison...")
    signal_comparison = analyzer.compare_signal_generation_metrics(requested_signal_metrics)
    
    print("Running classification metrics comparison...")
    classification_comparison = analyzer.enhanced_classification_comparison(requested_classification_metrics)
    
    # Run ablation study if baseline is specified
    if baseline:
        print("Running ablation study analysis...")
        ablation_results = analyzer.ablation_study_analysis()
        
        if ablation_results:
            print(f"Found {len(ablation_results)} ablation components")
            
            # Print top components
            print("\nTop ablation components by effect size:")
            for i, result in enumerate(ablation_results[:3]):
                print(f"  {i+1}. {result['component_name']}: {result['average_effect_size']:.3f}")
    
    # Generate comparison table
    print(f"\nGenerating comparison table in {args.format} format...")
    table_path = analyzer.generate_comparison_table(output_format=args.format)
    
    # Generate comprehensive report
    print("Generating comparative report...")
    report_path = analyzer.generate_comparative_report(output_dir)
    
    # Generate statistical summary
    print("Generating statistical summary...")
    statistical_summary = analyzer.generate_statistical_summary()
    
    # Print summary
    print("\n" + "="*50)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Total runs analyzed: {statistical_summary['total_runs']}")
    print(f"Baseline run: {statistical_summary['baseline_run'] or 'None'}")
    
    if 'comparison_summary' in statistical_summary:
        for comparison_type, summary in statistical_summary['comparison_summary'].items():
            print(f"\n{comparison_type.replace('_', ' ').title()}:")
            if 'total_comparisons' in summary:
                print(f"  Total comparisons: {summary['total_comparisons']}")
            if 'improvements' in summary:
                print(f"  Improvements: {summary['improvements']}")
            if 'degradations' in summary:
                print(f"  Degradations: {summary['degradations']}")
    
    if 'recommendations' in statistical_summary:
        print(f"\nRecommendations:")
        for i, rec in enumerate(statistical_summary['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nResults saved to:")
    print(f"  Comparison table: {table_path}")
    print(f"  Comparative report: {report_path}")
    
    # Generate performance rankings
    print("\nGenerating performance rankings...")
    rankings = analyzer.get_performance_rankings(metrics)
    
    for metric, ranking in rankings.items():
        if ranking:
            print(f"\n{metric.upper()} Rankings:")
            for i, rank_info in enumerate(ranking[:5]):  # Top 5
                print(f"  {rank_info['rank']}. {rank_info['run_name']}: {rank_info['value']:.4f}")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("ABR Transformer Evaluation Comparison Tool")
    print("=" * 50)
    
    # Expand glob patterns
    input_dirs = expand_glob_patterns(args.input_dirs)
    
    if not input_dirs:
        print("Error: No input directories found")
        sys.exit(1)
    
    # Validate input directories
    valid_dirs = validate_input_directories(input_dirs)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Group runs if requested
    if args.group_by:
        print(f"\nGrouping evaluation runs by {args.group_by}...")
        grouped_runs = group_evaluation_runs(valid_dirs, args.group_by)
        
        print(f"Found {len(grouped_runs)} groups:")
        for group_name, group_dirs in grouped_runs.items():
            print(f"  {group_name}: {len(group_dirs)} runs")
        
        # Run analysis for each group
        for group_name, group_dirs in grouped_runs.items():
            if len(group_dirs) < 2:
                print(f"\nSkipping group '{group_name}' - need at least 2 runs for comparison")
                continue
            
            print(f"\n{'='*30}")
            print(f"Analyzing group: {group_name}")
            print(f"{'='*30}")
            
            # Determine baseline for this group
            group_baseline = None
            if args.baseline:
                # Check if baseline is in this group
                baseline_path = Path(args.baseline)
                if any(Path(d) == baseline_path for d in group_dirs):
                    group_baseline = args.baseline
            
            # Create group-specific output directory
            group_output_dir = output_path / group_name
            group_output_dir.mkdir(exist_ok=True)
            
            # Run analysis for this group
            run_comparative_analysis(group_dirs, group_baseline, args.metrics, 
                                   str(group_output_dir), args)
    
    else:
        # Run single analysis for all directories
        run_comparative_analysis(valid_dirs, args.baseline, args.metrics, 
                               args.output_dir, args)
    
    print("\n" + "="*50)
    print("Comparative analysis completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
