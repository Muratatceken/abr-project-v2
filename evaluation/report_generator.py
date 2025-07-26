#!/usr/bin/env python3
"""
Report Generation for ABR Model Evaluation

This module provides comprehensive report generation including:
- CSV summary reports
- Clinical diagnostic reports
- PDF summary generation (optional)
- Alert and flag reports

Author: AI Assistant
Date: January 2025
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from datetime import datetime
import warnings

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """Comprehensive report generator for evaluation results."""
    
    def __init__(
        self,
        class_names: List[str] = None,
        output_dir: Union[str, Path] = "outputs/evaluation"
    ):
        """Initialize report generator.
        
        Args:
            class_names: List of class names
            output_dir: Output directory for reports
        """
        self.class_names = class_names or ["NORMAL", "NÖROPATİ", "SNİK", "TOTAL", "İTİK"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "alerts").mkdir(exist_ok=True)
    
    # ==================== CSV SUMMARY REPORTS ====================
    
    def generate_evaluation_summary_csv(
        self,
        metrics_dict: Dict[str, Dict[str, Any]],
        confidence_intervals: Dict[str, Dict[str, Tuple[float, float, float]]] = None,
        filename: str = "evaluation_summary.csv"
    ) -> Path:
        """
        Generate comprehensive evaluation summary CSV.
        
        Args:
            metrics_dict: Dictionary of computed metrics
            confidence_intervals: Bootstrap confidence intervals
            filename: Output filename
            
        Returns:
            Path to generated CSV file
        """
        csv_path = self.output_dir / "data" / filename
        
        # Prepare data for CSV
        csv_data = []
        
        for task_name, task_metrics in metrics_dict.items():
            for metric_name, metric_value in task_metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    row = {
                        'Task': task_name,
                        'Metric': metric_name,
                        'Value': float(metric_value),
                        'Units': self._get_metric_units(task_name, metric_name)
                    }
                    
                    # Add confidence intervals if available
                    if (confidence_intervals and 
                        task_name in confidence_intervals and
                        metric_name in confidence_intervals[task_name]):
                        
                        _, lower_ci, upper_ci = confidence_intervals[task_name][metric_name]
                        row['Lower_CI'] = float(lower_ci)
                        row['Upper_CI'] = float(upper_ci)
                    else:
                        row['Lower_CI'] = None
                        row['Upper_CI'] = None
                    
                    csv_data.append(row)
        
        # Write CSV
        if csv_data:
            fieldnames = ['Task', 'Metric', 'Value', 'Lower_CI', 'Upper_CI', 'Units']
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        return csv_path
    
    def generate_stratified_summary_csv(
        self,
        stratified_metrics: Dict[str, Dict[str, Any]],
        filename: str = "stratified_summary.csv"
    ) -> Path:
        """
        Generate stratified evaluation summary CSV.
        
        Args:
            stratified_metrics: Dictionary of stratified metrics
            filename: Output filename
            
        Returns:
            Path to generated CSV file
        """
        csv_path = self.output_dir / "data" / filename
        
        csv_data = []
        
        for stratify_by, strata_data in stratified_metrics.items():
            for stratum, stratum_metrics in strata_data.items():
                for metric_name, metric_value in stratum_metrics.items():
                    if isinstance(metric_value, (int, float, np.integer, np.floating)):
                        csv_data.append({
                            'Stratification': stratify_by,
                            'Stratum': stratum,
                            'Metric': metric_name,
                            'Value': float(metric_value),
                            'Units': self._get_metric_units_from_name(metric_name)
                        })
        
        # Write CSV
        if csv_data:
            fieldnames = ['Stratification', 'Stratum', 'Metric', 'Value', 'Units']
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        return csv_path
    
    def generate_per_sample_diagnostics_csv(
        self,
        sample_data: List[Dict[str, Any]],
        filename: str = "per_sample_diagnostics.csv"
    ) -> Path:
        """
        Generate per-sample diagnostic CSV.
        
        Args:
            sample_data: List of per-sample diagnostic data
            filename: Output filename
            
        Returns:
            Path to generated CSV file
        """
        csv_path = self.output_dir / "data" / filename
        
        if not sample_data:
            return csv_path
        
        # Extract all possible fieldnames
        fieldnames = set()
        for sample in sample_data:
            fieldnames.update(sample.keys())
        fieldnames = sorted(list(fieldnames))
        
        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_data)
        
        return csv_path
    
    # ==================== CLINICAL ERROR FLAGS ====================
    
    def generate_clinical_alerts(
        self,
        alert_data: List[Dict[str, Any]],
        filename: str = "clinical_alerts.json"
    ) -> Path:
        """
        Generate clinical alert report.
        
        Args:
            alert_data: List of clinical alerts
            filename: Output filename
            
        Returns:
            Path to generated JSON file
        """
        alerts_path = self.output_dir / "alerts" / filename
        
        # Add metadata
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_alerts': len(alert_data),
            'alert_summary': self._summarize_alerts(alert_data),
            'alerts': alert_data
        }
        
        with open(alerts_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return alerts_path
    
    def _summarize_alerts(self, alert_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize alert types and counts."""
        alert_summary = {}
        
        for alert in alert_data:
            if 'error_flags' in alert:
                for flag in alert['error_flags']:
                    alert_summary[flag] = alert_summary.get(flag, 0) + 1
        
        return alert_summary
    
    def identify_clinical_errors(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: Dict[str, np.ndarray],
        sample_ids: List[str] = None,
        thresholds: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify clinical errors and generate alerts.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth data
            sample_ids: Sample identifiers
            thresholds: Error thresholds
            
        Returns:
            List of clinical alerts
        """
        if thresholds is None:
            thresholds = {
                'threshold_error': 20.0,  # dB
                'peak_latency_error': 1.0,  # ms
                'peak_amplitude_error': 0.2  # μV
            }
        
        alerts = []
        n_samples = len(next(iter(ground_truth.values())))
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(n_samples)]
        
        for i in range(min(n_samples, len(sample_ids))):
            sample_alerts = []
            
            # Check threshold errors
            if 'threshold' in predictions and 'threshold' in ground_truth:
                pred_thresh = predictions['threshold'][i]
                true_thresh = ground_truth['threshold'][i]
                
                if isinstance(pred_thresh, (list, np.ndarray)):
                    pred_thresh = pred_thresh[0] if len(pred_thresh) > 0 else 0
                if isinstance(true_thresh, (list, np.ndarray)):
                    true_thresh = true_thresh[0] if len(true_thresh) > 0 else 0
                
                thresh_error = abs(float(pred_thresh) - float(true_thresh))
                
                if thresh_error > thresholds['threshold_error']:
                    if pred_thresh < true_thresh - thresholds['threshold_error']:
                        sample_alerts.append('false_clear')
                    elif pred_thresh > true_thresh + thresholds['threshold_error']:
                        sample_alerts.append('false_impairment')
                    else:
                        sample_alerts.append('threshold_error_>20')
            
            # Check peak detection errors
            if 'peak' in predictions and 'v_peak' in ground_truth and 'v_peak_mask' in ground_truth:
                # Peak existence
                if len(predictions['peak']) > 0:
                    pred_peak_exists = predictions['peak'][0][i] > 0.5
                else:
                    pred_peak_exists = False
                
                true_peak_exists = ground_truth['v_peak_mask'][i].any() if hasattr(ground_truth['v_peak_mask'][i], 'any') else bool(ground_truth['v_peak_mask'][i])
                
                if pred_peak_exists and not true_peak_exists:
                    sample_alerts.append('false_peak_detection')
                elif not pred_peak_exists and true_peak_exists:
                    sample_alerts.append('missed_peak')
                
                # Peak parameter errors (if both peaks exist)
                if pred_peak_exists and true_peak_exists:
                    # Latency error
                    if len(predictions['peak']) > 1:
                        pred_latency = predictions['peak'][1][i]
                        true_latency = ground_truth['v_peak'][i][0] if len(ground_truth['v_peak'][i]) > 0 else 0
                        
                        if abs(float(pred_latency) - float(true_latency)) > thresholds['peak_latency_error']:
                            sample_alerts.append('peak_latency_error')
                    
                    # Amplitude error
                    if len(predictions['peak']) > 2:
                        pred_amplitude = predictions['peak'][2][i]
                        true_amplitude = ground_truth['v_peak'][i][1] if len(ground_truth['v_peak'][i]) > 1 else 0
                        
                        if abs(float(pred_amplitude) - float(true_amplitude)) > thresholds['peak_amplitude_error']:
                            sample_alerts.append('peak_amplitude_error')
            
            # Add alert if any errors found
            if sample_alerts:
                alerts.append({
                    'sample_id': sample_ids[i],
                    'error_flags': sample_alerts,
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    # ==================== PDF REPORT GENERATION ====================
    
    def generate_pdf_summary(
        self,
        metrics_dict: Dict[str, Dict[str, Any]],
        figures_dir: Path = None,
        filename: str = "evaluation_summary.pdf"
    ) -> Optional[Path]:
        """
        Generate PDF summary report (requires reportlab).
        
        Args:
            metrics_dict: Dictionary of computed metrics
            figures_dir: Directory containing figures to include
            filename: Output filename
            
        Returns:
            Path to generated PDF file or None if reportlab not available
        """
        if not REPORTLAB_AVAILABLE:
            warnings.warn("reportlab not available. PDF generation skipped.")
            return None
        
        pdf_path = self.output_dir / "reports" / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("ABR Model Evaluation Summary", title_style))
        story.append(Spacer(1, 20))
        
        # Generation info
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Metrics summary table
        story.append(Paragraph("Performance Metrics Summary", styles['Heading2']))
        
        # Prepare table data
        table_data = [['Task', 'Metric', 'Value', 'Units']]
        
        for task_name, task_metrics in metrics_dict.items():
            for metric_name, metric_value in task_metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    units = self._get_metric_units(task_name, metric_name)
                    table_data.append([
                        task_name,
                        metric_name,
                        f"{float(metric_value):.4f}",
                        units
                    ])
        
        # Create table
        if len(table_data) > 1:
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        # Build PDF
        doc.build(story)
        return pdf_path
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def _get_metric_units(self, task_name: str, metric_name: str) -> str:
        """Get appropriate units for a metric."""
        units_map = {
            'mse': 'μV²',
            'mae': 'μV',
            'rmse': 'μV',
            'snr': 'dB',
            'correlation': '',
            'r2': '',
            'accuracy': '',
            'precision': '',
            'recall': '',
            'f1': '',
            'latency_mae': 'ms',
            'latency_mse': 'ms²',
            'latency_rmse': 'ms',
            'amplitude_mae': 'μV',
            'amplitude_mse': 'μV²',
            'amplitude_rmse': 'μV',
        }
        
        # Check for threshold-related metrics
        if 'threshold' in metric_name.lower():
            if 'mae' in metric_name or 'mse' in metric_name or 'rmse' in metric_name or 'error' in metric_name:
                return 'dB HL'
        
        # Check for latency-related metrics
        if 'latency' in metric_name.lower():
            return 'ms'
        
        # Check for amplitude-related metrics
        if 'amplitude' in metric_name.lower():
            return 'μV'
        
        return units_map.get(metric_name.lower(), '')
    
    def _get_metric_units_from_name(self, metric_name: str) -> str:
        """Get units from metric name only."""
        return self._get_metric_units('', metric_name)
    
    def create_summary_table_text(
        self,
        metrics_dict: Dict[str, Dict[str, Any]],
        confidence_intervals: Dict[str, Dict[str, Tuple[float, float, float]]] = None
    ) -> str:
        """
        Create a formatted text summary table.
        
        Args:
            metrics_dict: Dictionary of computed metrics
            confidence_intervals: Bootstrap confidence intervals
            
        Returns:
            Formatted text table
        """
        lines = []
        lines.append("🔬 COMPREHENSIVE ABR MODEL EVALUATION SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        for task_name, task_metrics in metrics_dict.items():
            lines.append(f"📊 {task_name.upper()}:")
            
            for metric_name, metric_value in task_metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    units = self._get_metric_units(task_name, metric_name)
                    
                    # Format the metric line
                    metric_line = f"   {metric_name}: {float(metric_value):.4f}"
                    
                    # Add confidence intervals if available
                    if (confidence_intervals and 
                        task_name in confidence_intervals and
                        metric_name in confidence_intervals[task_name]):
                        
                        _, lower_ci, upper_ci = confidence_intervals[task_name][metric_name]
                        metric_line += f" [{lower_ci:.4f}, {upper_ci:.4f}]"
                    
                    if units:
                        metric_line += f" {units}"
                    
                    lines.append(metric_line)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_configuration(
        self,
        config: Dict[str, Any],
        filename: str = "evaluation_config.json"
    ) -> Path:
        """
        Save evaluation configuration.
        
        Args:
            config: Configuration dictionary
            filename: Output filename
            
        Returns:
            Path to saved configuration file
        """
        config_path = self.output_dir / "data" / filename
        
        # Add metadata
        config_with_meta = {
            'generated_at': datetime.now().isoformat(),
            'configuration': config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_with_meta, f, indent=2, ensure_ascii=False, default=str)
        
        return config_path 