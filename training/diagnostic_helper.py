#!/usr/bin/env python3
"""
ABR Training Diagnostic Helper

This script provides diagnostic tools to analyze training issues and suggest improvements.
It can analyze loss patterns, class imbalance, learning rates, and other training dynamics.

Author: AI Assistant
Date: January 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABRTrainingDiagnostic:
    """Comprehensive diagnostic tool for ABR training analysis."""
    
    def __init__(self, log_dir: str = "outputs/production"):
        self.log_dir = Path(log_dir)
        self.training_logs = []
        self.config = {}
        
    def load_training_data(self) -> bool:
        """Load training logs and configuration."""
        try:
            # Load config
            config_path = self.log_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
            
            # Load training logs (if available)
            log_path = self.log_dir / "training.log"
            if log_path.exists():
                self.parse_training_logs(log_path)
                logger.info(f"Loaded training logs from {log_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False
    
    def parse_training_logs(self, log_path: Path):
        """Parse training logs to extract metrics."""
        with open(log_path, 'r') as f:
            for line in f:
                if "Epoch" in line and "Train Loss" in line:
                    # Parse epoch metrics
                    parts = line.split('|')
                    if len(parts) >= 4:
                        epoch_part = parts[0].strip()
                        train_part = parts[1].strip()
                        val_part = parts[2].strip()
                        f1_part = parts[3].strip()
                        
                        try:
                            epoch = int(epoch_part.split()[-1])
                            train_loss = float(train_part.split(':')[-1].strip())
                            val_loss = float(val_part.split(':')[-1].strip())
                            f1_score = float(f1_part.split(':')[-1].strip())
                            
                            self.training_logs.append({
                                'epoch': epoch,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'f1_score': f1_score
                            })
                        except (ValueError, IndexError):
                            continue
    
    def analyze_training_issues(self) -> Dict[str, Any]:
        """Analyze training logs and identify potential issues."""
        issues = {
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'metrics_analysis': {}
        }
        
        if not self.training_logs:
            issues['critical_issues'].append("No training logs found for analysis")
            return issues
        
        # Analyze F1 scores
        f1_scores = [log['f1_score'] for log in self.training_logs]
        if all(f1 == 0.0 for f1 in f1_scores):
            issues['critical_issues'].append(
                "F1 scores are consistently 0.0 - indicates classification failure"
            )
            issues['recommendations'].extend([
                "Check if classification logits are being computed correctly",
                "Verify loss function is including classification loss",
                "Check if targets are in correct format (LongTensor)",
                "Ensure model outputs include 'classification_logits' key"
            ])
        
        # Analyze loss progression
        train_losses = [log['train_loss'] for log in self.training_logs]
        val_losses = [log['val_loss'] for log in self.training_logs]
        
        if len(train_losses) > 5:
            # Check for loss plateau
            recent_train = train_losses[-5:]
            if max(recent_train) - min(recent_train) < 0.1:
                issues['warnings'].append("Training loss has plateaued")
                issues['recommendations'].append("Consider reducing learning rate or adjusting loss weights")
            
            # Check for overfitting
            if len(val_losses) > 5:
                recent_val = val_losses[-5:]
                if any(v > t for v, t in zip(recent_val, recent_train[-5:])):
                    gap = np.mean(recent_val) - np.mean(recent_train)
                    if gap > 1.0:
                        issues['warnings'].append(f"Potential overfitting detected (val-train gap: {gap:.3f})")
                        issues['recommendations'].extend([
                            "Consider increasing regularization",
                            "Add more data augmentation",
                            "Reduce model complexity"
                        ])
        
        # Analyze loss magnitude
        if train_losses:
            avg_train_loss = np.mean(train_losses)
            if avg_train_loss > 10.0:
                issues['warnings'].append(f"High training loss (avg: {avg_train_loss:.3f})")
                issues['recommendations'].extend([
                    "Check loss scaling - may need to adjust loss component weights",
                    "Verify input normalization",
                    "Consider reducing learning rate"
                ])
        
        # Store metrics analysis
        issues['metrics_analysis'] = {
            'avg_train_loss': np.mean(train_losses) if train_losses else 0,
            'avg_val_loss': np.mean(val_losses) if val_losses else 0,
            'avg_f1_score': np.mean(f1_scores) if f1_scores else 0,
            'loss_std': np.std(train_losses) if train_losses else 0,
            'epochs_analyzed': len(self.training_logs)
        }
        
        return issues
    
    def suggest_improvements(self) -> Dict[str, List[str]]:
        """Suggest specific improvements based on analysis."""
        improvements = {
            'immediate_fixes': [],
            'hyperparameter_tuning': [],
            'architectural_changes': [],
            'data_improvements': []
        }
        
        # Analyze current config
        if self.config:
            # Learning rate analysis
            lr = self.config.get('learning_rate', 1e-4)
            if lr > 1e-3:
                improvements['hyperparameter_tuning'].append(
                    f"Learning rate ({lr}) may be too high, try reducing to 1e-4 or lower"
                )
            
            # Batch size analysis
            batch_size = self.config.get('batch_size', 32)
            if batch_size < 16:
                improvements['hyperparameter_tuning'].append(
                    "Small batch size may cause unstable training, consider increasing"
                )
            
            # Loss weights analysis
            loss_config = self.config.get('loss', {})
            loss_weights = loss_config.get('loss_weights', {})
            
            if loss_weights.get('classification', 1.0) < 1.0:
                improvements['immediate_fixes'].append(
                    "Classification loss weight is low - increase to improve F1 scores"
                )
        
        # General recommendations based on common issues
        improvements['immediate_fixes'].extend([
            "Add detailed loss component logging to monitor signal/peak/classification losses separately",
            "Implement proper F1 score calculation in validation loop",
            "Add classification report logging every few epochs"
        ])
        
        improvements['hyperparameter_tuning'].extend([
            "Try curriculum learning with gradual loss weight ramping",
            "Experiment with different optimizers (AdamW with different betas)",
            "Consider cosine annealing with warm restarts for learning rate"
        ])
        
        improvements['architectural_changes'].extend([
            "Add skip connections in classification head",
            "Implement multi-scale feature extraction",
            "Consider attention mechanisms for better feature selection"
        ])
        
        improvements['data_improvements'].extend([
            "Analyze class distribution and implement better balancing strategies",
            "Add more sophisticated data augmentation",
            "Consider synthetic data generation for minority classes"
        ])
        
        return improvements
    
    def create_diagnostic_plots(self, save_dir: str = "diagnostic_plots") -> List[str]:
        """Create diagnostic plots for training analysis."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        created_plots = []
        
        if not self.training_logs:
            logger.warning("No training logs available for plotting")
            return created_plots
        
        # Extract data
        epochs = [log['epoch'] for log in self.training_logs]
        train_losses = [log['train_loss'] for log in self.training_logs]
        val_losses = [log['val_loss'] for log in self.training_logs]
        f1_scores = [log['f1_score'] for log in self.training_logs]
        
        # Plot 1: Loss curves
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(epochs, val_losses, label='Val Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: F1 Score progression
        plt.subplot(2, 2, 2)
        plt.plot(epochs, f1_scores, label='F1 Score', color='green', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Loss ratio analysis
        plt.subplot(2, 2, 3)
        if len(train_losses) > 1 and len(val_losses) > 1:
            loss_ratios = [v/t if t > 0 else 1 for v, t in zip(val_losses, train_losses)]
            plt.plot(epochs, loss_ratios, label='Val/Train Ratio', color='orange', alpha=0.7)
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Ratio')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Ratio')
            plt.title('Validation/Training Loss Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Loss smoothing analysis
        plt.subplot(2, 2, 4)
        if len(train_losses) > 5:
            # Simple moving average
            window = min(5, len(train_losses) // 2)
            train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(val_losses[:len(train_smooth)], np.ones(window)/window, mode='valid')
            smooth_epochs = epochs[:len(train_smooth)]
            
            plt.plot(smooth_epochs, train_smooth, label='Train (Smoothed)', color='darkblue')
            plt.plot(smooth_epochs, val_smooth, label='Val (Smoothed)', color='darkred')
            plt.xlabel('Epoch')
            plt.ylabel('Smoothed Loss')
            plt.title('Smoothed Loss Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_path / "training_diagnostic.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_plots.append(str(plot_path))
        
        logger.info(f"Created diagnostic plots in {save_path}")
        return created_plots
    
    def generate_report(self, output_file: str = "training_diagnostic_report.md") -> str:
        """Generate a comprehensive diagnostic report."""
        issues = self.analyze_training_issues()
        improvements = self.suggest_improvements()
        
        report_lines = [
            "# ABR Training Diagnostic Report",
            f"Generated on: {np.datetime64('now')}",
            "",
            "## Training Configuration Analysis",
            ""
        ]
        
        if self.config:
            report_lines.extend([
                f"- **Learning Rate**: {self.config.get('learning_rate', 'N/A')}",
                f"- **Batch Size**: {self.config.get('batch_size', 'N/A')}",
                f"- **Epochs**: {self.config.get('num_epochs', 'N/A')}",
                f"- **Mixed Precision**: {self.config.get('use_amp', 'N/A')}",
                f"- **Optimizer**: {self.config.get('optimizer', {}).get('type', 'N/A')}",
                ""
            ])
        
        # Metrics Analysis
        if issues['metrics_analysis']:
            metrics = issues['metrics_analysis']
            report_lines.extend([
                "## Metrics Analysis",
                "",
                f"- **Average Training Loss**: {metrics.get('avg_train_loss', 0):.4f}",
                f"- **Average Validation Loss**: {metrics.get('avg_val_loss', 0):.4f}",
                f"- **Average F1 Score**: {metrics.get('avg_f1_score', 0):.4f}",
                f"- **Loss Standard Deviation**: {metrics.get('loss_std', 0):.4f}",
                f"- **Epochs Analyzed**: {metrics.get('epochs_analyzed', 0)}",
                ""
            ])
        
        # Critical Issues
        if issues['critical_issues']:
            report_lines.extend([
                "## 🚨 Critical Issues",
                ""
            ])
            for issue in issues['critical_issues']:
                report_lines.append(f"- **{issue}**")
            report_lines.append("")
        
        # Warnings
        if issues['warnings']:
            report_lines.extend([
                "## ⚠️ Warnings",
                ""
            ])
            for warning in issues['warnings']:
                report_lines.append(f"- {warning}")
            report_lines.append("")
        
        # Recommendations
        if issues['recommendations']:
            report_lines.extend([
                "## 💡 Immediate Recommendations",
                ""
            ])
            for rec in issues['recommendations']:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        # Improvement Suggestions
        for category, suggestions in improvements.items():
            if suggestions:
                category_name = category.replace('_', ' ').title()
                report_lines.extend([
                    f"## 🔧 {category_name}",
                    ""
                ])
                for suggestion in suggestions:
                    report_lines.append(f"- {suggestion}")
                report_lines.append("")
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated diagnostic report: {output_file}")
        return output_file


def main():
    """Run diagnostic analysis on ABR training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ABR Training Diagnostic Tool')
    parser.add_argument('--log-dir', type=str, default='outputs/production',
                       help='Directory containing training logs and config')
    parser.add_argument('--output-dir', type=str, default='diagnostic_output',
                       help='Directory to save diagnostic output')
    parser.add_argument('--create-plots', action='store_true', default=True,
                       help='Create diagnostic plots')
    
    args = parser.parse_args()
    
    # Create diagnostic instance
    diagnostic = ABRTrainingDiagnostic(args.log_dir)
    
    # Load training data
    if not diagnostic.load_training_data():
        logger.error("Failed to load training data")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate report
    report_file = output_dir / "diagnostic_report.md"
    diagnostic.generate_report(str(report_file))
    
    # Create plots if requested
    if args.create_plots:
        plot_dir = output_dir / "plots"
        diagnostic.create_diagnostic_plots(str(plot_dir))
    
    print(f"\n✅ Diagnostic analysis complete!")
    print(f"📊 Report saved to: {report_file}")
    if args.create_plots:
        print(f"📈 Plots saved to: {plot_dir}")


if __name__ == '__main__':
    main() 