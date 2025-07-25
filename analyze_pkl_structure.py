#!/usr/bin/env python3
"""
PKL Structure Analysis Module

This module contains all analysis functions for examining PKL file structures,
denoising effectiveness, and dataset statistics. These functions are separated
from preprocessing.py to maintain clean separation of concerns.

Functions:
- analyze_complete_dataset_denoising(): Complete dataset denoising analysis
- estimate_fmp_before_denoising(): FMP estimation for original signals
- print_analysis_summary(): Results reporting
- create_comprehensive_plots(): Visual analysis generation
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict
import warnings
import os
from utils.preprocessing import denoise, estimate_fmp_after_denoising

warnings.filterwarnings('ignore')

def analyze_complete_dataset_denoising(
    input_file: str = "data/processed/processed_data.pkl",
    sample_size: int = 1000,
    verbose: bool = True
) -> dict:
    """
    Analyze denoising effectiveness on the complete categorical dataset without FMP filtering.
    
    Args:
        input_file (str): Path to complete categorical dataset
        sample_size (int): Number of samples to analyze (default: 1000)
        verbose (bool): If True, print detailed information
        
    Returns:
        dict: Comprehensive analysis results
    """
    if verbose:
        print(f"🔄 Analyzing denoising on complete categorical dataset...")
        print(f"   Input: {input_file}")
        print(f"   Sample size: {sample_size}")
        print()
    
    # Load complete categorical dataset
    try:
        data = joblib.load(input_file)
        label_encoder = joblib.load('data/processed/label_encoder.pkl')
        if verbose:
            print(f"✅ Loaded {len(data)} samples from complete categorical dataset")
            print(f"✅ Loaded label encoder with categories: {label_encoder.classes_}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e}")
    
    # Analyze dataset composition
    samples_by_type = defaultdict(list)
    for i, sample in enumerate(data):
        hearing_loss_type = int(sample['static_params'][5])
        samples_by_type[hearing_loss_type].append((i, sample))
    
    if verbose:
        print(f"\n📊 Dataset composition (complete, no FMP filtering):")
        for category_id in sorted(samples_by_type.keys()):
            category_name = label_encoder.classes_[category_id - 1]
            count = len(samples_by_type[category_id])
            print(f"   {category_name}: {count:,} samples")
    
    # Sample for analysis
    np.random.seed(42)
    total_samples = len(data)
    actual_sample_size = min(sample_size, total_samples)
    indices = np.random.choice(total_samples, actual_sample_size, replace=False)
    
    if verbose:
        print(f"\n🔬 Analyzing {actual_sample_size} samples for denoising effectiveness...")
    
    # Initialize analysis containers
    analysis_data = defaultdict(lambda: {
        'fmp_before': [],
        'fmp_after': [],
        'improvement_ratio': [],
        'correlation': [],
        'snr_improvement': [],
        'energy_retention': [],
        'noise_reduction': [],
        'patient_ids': []
    })
    
    all_data = {
        'fmp_before': [],
        'fmp_after': [],
        'improvement_ratio': [],
        'correlation': [],
        'snr_improvement': [],
        'energy_retention': [],
        'noise_reduction': [],
        'hearing_loss_type': []
    }
    
    # Process samples
    for i, idx in enumerate(indices):
        if verbose and i % 100 == 0:
            print(f"   Processing {i}/{actual_sample_size}...")
        
        sample = data[idx]
        original_signal = sample['signal']
        hearing_loss_type = int(sample['static_params'][5])
        patient_id = sample['patient_id']
        
        try:
            # Estimate FMP before denoising
            fmp_before = estimate_fmp_before_denoising(original_signal)
            
            # Apply denoising
            denoised_signal = denoise(original_signal, fs=20000, wavelet='db4', level=3)
            
            # Calculate FMP after denoising
            fmp_after = estimate_fmp_after_denoising(original_signal, denoised_signal)
            
            # Handle invalid values
            if np.isnan(fmp_after) or np.isinf(fmp_after):
                fmp_after = 0.0
            if np.isnan(fmp_before) or np.isinf(fmp_before):
                fmp_before = 0.1
            
            # Calculate correlation
            correlation = np.corrcoef(original_signal, denoised_signal)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Calculate SNR improvement
            noise_estimate = original_signal - denoised_signal
            signal_power = np.var(denoised_signal)
            noise_power = np.var(noise_estimate)
            snr_improvement = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
            
            # Calculate energy retention
            orig_energy = np.sum(original_signal**2)
            den_energy = np.sum(denoised_signal**2)
            energy_retention = den_energy / orig_energy if orig_energy > 0 else 0
            
            # Calculate noise reduction percentage
            orig_noise_level = np.std(original_signal)
            remaining_noise_level = np.std(noise_estimate)
            noise_reduction = (1 - remaining_noise_level / orig_noise_level) * 100 if orig_noise_level > 0 else 0
            
            # Calculate improvement ratio
            improvement_ratio = fmp_after / fmp_before if fmp_before > 0 else 1.0
            
            # Store data by category
            analysis_data[hearing_loss_type]['fmp_before'].append(fmp_before)
            analysis_data[hearing_loss_type]['fmp_after'].append(fmp_after)
            analysis_data[hearing_loss_type]['improvement_ratio'].append(improvement_ratio)
            analysis_data[hearing_loss_type]['correlation'].append(correlation)
            analysis_data[hearing_loss_type]['snr_improvement'].append(snr_improvement)
            analysis_data[hearing_loss_type]['energy_retention'].append(energy_retention)
            analysis_data[hearing_loss_type]['noise_reduction'].append(noise_reduction)
            analysis_data[hearing_loss_type]['patient_ids'].append(patient_id)
            
            # Store overall data
            all_data['fmp_before'].append(fmp_before)
            all_data['fmp_after'].append(fmp_after)
            all_data['improvement_ratio'].append(improvement_ratio)
            all_data['correlation'].append(correlation)
            all_data['snr_improvement'].append(snr_improvement)
            all_data['energy_retention'].append(energy_retention)
            all_data['noise_reduction'].append(noise_reduction)
            all_data['hearing_loss_type'].append(hearing_loss_type)
            
        except Exception as e:
            if verbose:
                print(f"     Warning: Error processing sample {idx}: {e}")
            continue
    
    # Calculate comprehensive statistics
    results = {
        'dataset_info': {
            'total_samples': total_samples,
            'analyzed_samples': len(all_data['fmp_before']),
            'sample_percentage': len(all_data['fmp_before']) / total_samples * 100,
            'no_fmp_filtering': True,
            'categories': {cat_id: len(samples_by_type[cat_id]) for cat_id in samples_by_type.keys()}
        },
        'overall_stats': {
            'fmp_before_mean': np.mean(all_data['fmp_before']),
            'fmp_before_std': np.std(all_data['fmp_before']),
            'fmp_after_mean': np.mean(all_data['fmp_after']),
            'fmp_after_std': np.std(all_data['fmp_after']),
            'improvement_ratio_mean': np.mean(all_data['improvement_ratio']),
            'improvement_ratio_std': np.std(all_data['improvement_ratio']),
            'improved_samples': sum(1 for ratio in all_data['improvement_ratio'] if ratio > 1.0),
            'improvement_rate': sum(1 for ratio in all_data['improvement_ratio'] if ratio > 1.0) / len(all_data['improvement_ratio']) * 100,
            'correlation_mean': np.mean(all_data['correlation']),
            'snr_improvement_mean': np.mean(all_data['snr_improvement']),
            'energy_retention_mean': np.mean(all_data['energy_retention']),
            'noise_reduction_mean': np.mean(all_data['noise_reduction'])
        },
        'category_stats': {},
        'raw_data': {
            'analysis_data': dict(analysis_data),
            'all_data': all_data,
            'category_names': label_encoder.classes_
        }
    }
    
    # Calculate category-specific statistics
    for category_id in sorted(analysis_data.keys()):
        category_name = label_encoder.classes_[category_id - 1]
        cat_data = analysis_data[category_id]
        
        if cat_data['fmp_before']:
            results['category_stats'][category_id] = {
                'name': category_name,
                'samples': len(cat_data['fmp_before']),
                'fmp_before_mean': np.mean(cat_data['fmp_before']),
                'fmp_after_mean': np.mean(cat_data['fmp_after']),
                'improvement_ratio_mean': np.mean(cat_data['improvement_ratio']),
                'improved_count': sum(1 for ratio in cat_data['improvement_ratio'] if ratio > 1.0),
                'improvement_rate': sum(1 for ratio in cat_data['improvement_ratio'] if ratio > 1.0) / len(cat_data['improvement_ratio']) * 100,
                'correlation_mean': np.mean(cat_data['correlation']),
                'snr_improvement_mean': np.mean(cat_data['snr_improvement']),
                'energy_retention_mean': np.mean(cat_data['energy_retention']),
                'noise_reduction_mean': np.mean(cat_data['noise_reduction'])
            }
    
    if verbose:
        print_analysis_summary(results)
    
    return results

def estimate_fmp_before_denoising(signal):
    """
    Estimate FMP for the original signal using frequency domain analysis.
    """
    fs = 20000
    
    # Compute FFT
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft)
    
    # Define frequency ranges
    abr_freq_mask = (np.abs(freqs) >= 100) & (np.abs(freqs) <= 1500)  # ABR relevant frequencies
    noise_freq_mask = np.abs(freqs) > 1500  # High frequency noise
    
    # Calculate power in each range
    abr_power = np.sum(magnitude[abr_freq_mask]**2)
    noise_power = np.sum(magnitude[noise_freq_mask]**2)
    
    # Calculate FMP as ratio of ABR signal power to noise power
    if noise_power > 0:
        fmp_before = abr_power / noise_power
    else:
        fmp_before = abr_power  # If no noise detected, use signal power
    
    # Normalize to reasonable range
    fmp_before = fmp_before / len(signal)
    
    return fmp_before if not np.isnan(fmp_before) and not np.isinf(fmp_before) else 0.1

def print_analysis_summary(results):
    """Print a summary of the denoising analysis results."""
    print(f"\n📈 COMPLETE DATASET DENOISING ANALYSIS RESULTS")
    print(f"=" * 60)
    
    dataset_info = results['dataset_info']
    overall = results['overall_stats']
    
    print(f"📊 Dataset Information:")
    print(f"   Total samples in dataset: {dataset_info['total_samples']:,}")
    print(f"   Samples analyzed: {dataset_info['analyzed_samples']:,} ({dataset_info['sample_percentage']:.1f}%)")
    print(f"   FMP filtering: {'None - Complete dataset' if dataset_info['no_fmp_filtering'] else 'Applied'}")
    
    print(f"\n📈 Overall Denoising Performance:")
    print(f"   FMP before: {overall['fmp_before_mean']:.3f} ± {overall['fmp_before_std']:.3f}")
    print(f"   FMP after: {overall['fmp_after_mean']:.3f} ± {overall['fmp_after_std']:.3f}")
    print(f"   Average improvement ratio: {overall['improvement_ratio_mean']:.2f}x")
    print(f"   Samples with improvement: {overall['improved_samples']:,}/{dataset_info['analyzed_samples']:,} ({overall['improvement_rate']:.1f}%)")
    print(f"   Average correlation: {overall['correlation_mean']:.3f}")
    print(f"   Average SNR improvement: {overall['snr_improvement_mean']:.1f} dB")
    print(f"   Average energy retention: {overall['energy_retention_mean']:.1f}%")
    print(f"   Average noise reduction: {overall['noise_reduction_mean']:.1f}%")
    
    print(f"\n📊 Performance by Hearing Loss Type:")
    for category_id, stats in results['category_stats'].items():
        print(f"   {stats['name']}:")
        print(f"     Samples: {stats['samples']:,}")
        print(f"     Avg improvement ratio: {stats['improvement_ratio_mean']:.2f}x")
        print(f"     Success rate: {stats['improved_count']}/{stats['samples']} ({stats['improvement_rate']:.1f}%)")
        print(f"     Correlation: {stats['correlation_mean']:.3f}")
        print(f"     Noise reduction: {stats['noise_reduction_mean']:.1f}%")

def analyze_model_ready_dataset(
    input_file: str = "data/processed/model_ready_dataset.pkl",
    verbose: bool = True
) -> dict:
    """
    Analyze the model-ready dataset structure and statistics.
    
    Args:
        input_file (str): Path to model-ready dataset
        verbose (bool): If True, print detailed information
        
    Returns:
        dict: Dataset analysis results
    """
    if verbose:
        print(f"🔄 Analyzing model-ready dataset...")
        print(f"   Input: {input_file}")
        print()
    
    # Load model-ready dataset
    try:
        data = joblib.load(input_file)
        if verbose:
            print(f"✅ Loaded {len(data)} samples from model-ready dataset")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model-ready dataset not found: {e}")
    
    # Analyze dataset structure
    if not data:
        print("❌ Dataset is empty!")
        return {}
    
    sample = data[0]
    
    if verbose:
        print(f"\n📊 Dataset Structure Analysis:")
        print(f"   Total samples: {len(data):,}")
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Static parameters shape: {sample['static_params'].shape}")
        print(f"   Signal shape: {sample['signal'].shape}")
        print(f"   V peak shape: {sample['v_peak'].shape}")
        print(f"   Target shape: {sample['target'].shape}")
    
    # Analyze data distributions
    static_params = np.array([s['static_params'] for s in data])
    signals = np.array([s['signal'] for s in data])
    v_peaks = np.array([s['v_peak'] for s in data])
    targets = np.array([s['target'] for s in data])
    
    # Count by hearing loss type
    target_values = [t[0] if isinstance(t, np.ndarray) else t for t in targets]
    target_counts = np.bincount(np.array(target_values).astype(int))
    
    if verbose:
        print(f"\n📈 Data Statistics:")
        print(f"   Static parameters:")
        print(f"     Age - Mean: {static_params[:, 0].mean():.3f}, Std: {static_params[:, 0].std():.3f}")
        print(f"     Intensity - Mean: {static_params[:, 1].mean():.3f}, Std: {static_params[:, 1].std():.3f}")
        print(f"     Stimulus Rate - Mean: {static_params[:, 2].mean():.3f}, Std: {static_params[:, 2].std():.3f}")
        
        print(f"   Signals:")
        print(f"     Mean amplitude: {signals.mean():.6f}")
        print(f"     Std amplitude: {signals.std():.6f}")
        print(f"     Min amplitude: {signals.min():.6f}")
        print(f"     Max amplitude: {signals.max():.6f}")
        
        print(f"   V Peaks:")
        valid_v_peaks = v_peaks[~np.isnan(v_peaks)]
        if len(valid_v_peaks) > 0:
            print(f"     Valid V peaks: {len(valid_v_peaks)}/{len(v_peaks)} ({len(valid_v_peaks)/len(v_peaks)*100:.1f}%)")
            print(f"     V peak mean: {valid_v_peaks.mean():.3f}")
            print(f"     V peak std: {valid_v_peaks.std():.3f}")
        
        print(f"   Target Distribution:")
        for i, count in enumerate(target_counts):
            if count > 0:
                print(f"     Class {i}: {count:,} samples ({count/len(data)*100:.1f}%)")
    
    results = {
        'total_samples': len(data),
        'static_params_stats': {
            'age_mean': static_params[:, 0].mean(),
            'age_std': static_params[:, 0].std(),
            'intensity_mean': static_params[:, 1].mean(),
            'intensity_std': static_params[:, 1].std(),
            'rate_mean': static_params[:, 2].mean(),
            'rate_std': static_params[:, 2].std()
        },
        'signal_stats': {
            'mean': signals.mean(),
            'std': signals.std(),
            'min': signals.min(),
            'max': signals.max()
        },
        'v_peak_stats': {
            'valid_count': len(valid_v_peaks) if len(valid_v_peaks) > 0 else 0,
            'valid_percentage': len(valid_v_peaks)/len(v_peaks)*100 if len(valid_v_peaks) > 0 else 0,
            'mean': valid_v_peaks.mean() if len(valid_v_peaks) > 0 else 0,
            'std': valid_v_peaks.std() if len(valid_v_peaks) > 0 else 0
        },
        'target_distribution': {i: int(count) for i, count in enumerate(target_counts) if count > 0}
    }
    
    return results

if __name__ == "__main__":
    print("📊 PKL Structure Analysis Module")
    print("=" * 40)
    print("Available functions:")
    print("- analyze_complete_dataset_denoising()")
    print("- analyze_model_ready_dataset()")
    print("- estimate_fmp_before_denoising()")
    print("- print_analysis_summary()")
    print("\nImport this module to use analysis functions.") 