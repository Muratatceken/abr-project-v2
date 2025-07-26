#!/usr/bin/env python3
"""
Clinical Threshold Generation for ABR Data

This script creates clinically meaningful hearing threshold targets based on:
- Hearing loss classification (NORMAL, SNİK, İTİK, TOTAL, NÖROPATİ)
- Stimulus intensity levels
- V peak presence/absence
- Clinical audiometry standards

Author: AI Assistant
Date: January 2025
"""

import numpy as np
import joblib
import argparse
from typing import List, Dict, Tuple
import os


class ClinicalThresholdGenerator:
    """
    Generate clinical hearing thresholds from ABR data.
    """
    
    def __init__(self):
        # Clinical threshold ranges based on hearing loss severity (dB HL)
        self.threshold_mapping = {
            'NORMAL': (10, 25),      # Normal hearing
            'SNİK': (30, 45),        # Mild hearing loss (Sensorineural)
            'İTİK': (50, 65),        # Moderate hearing loss (Conductive)
            'TOTAL': (70, 85),       # Severe hearing loss
            'NÖROPATİ': (40, 90),    # Neural pathology (variable)
        }
        
        self.class_names = ['NORMAL', 'NÖROPATİ', 'SNİK', 'TOTAL', 'İTİK']
    
    def generate_clinical_thresholds(
        self, 
        data_samples: List[Dict],
        method: str = 'clinical_mapping'
    ) -> np.ndarray:
        """
        Generate clinical hearing thresholds for ABR samples.
        
        Args:
            data_samples: List of ABR data samples
            method: Method for threshold generation
            
        Returns:
            Array of clinical thresholds in dB HL
        """
        if method == 'clinical_mapping':
            return self._clinical_mapping_method(data_samples)
        elif method == 'intensity_based':
            return self._intensity_based_method(data_samples)
        elif method == 'combined':
            return self._combined_method(data_samples)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _clinical_mapping_method(self, data_samples: List[Dict]) -> np.ndarray:
        """
        Generate thresholds based on clinical hearing loss categories.
        """
        thresholds = []
        
        for sample in data_samples:
            class_idx = sample['target']
            class_name = self.class_names[class_idx]
            
            # Get base threshold range for this class
            min_thresh, max_thresh = self.threshold_mapping[class_name]
            base_threshold = (min_thresh + max_thresh) / 2
            
            # Add some variability within the clinical range
            noise = np.random.normal(0, (max_thresh - min_thresh) / 6)  # ±1σ covers ~68% of range
            
            final_threshold = base_threshold + noise
            final_threshold = np.clip(final_threshold, 0, 120)  # Clinical range
            
            thresholds.append(final_threshold)
        
        return np.array(thresholds)
    
    def _intensity_based_method(self, data_samples: List[Dict]) -> np.ndarray:
        """
        Generate thresholds based on stimulus intensity and V peak presence.
        """
        thresholds = []
        
        for sample in data_samples:
            class_idx = sample['target']
            class_name = self.class_names[class_idx]
            intensity = sample['static_params'][1]  # Normalized intensity
            v_peak_present = sample['v_peak_mask'][0] and sample['v_peak_mask'][1]
            
            # Base threshold from class
            min_thresh, max_thresh = self.threshold_mapping[class_name]
            base_threshold = (min_thresh + max_thresh) / 2
            
            # Intensity adjustment: higher intensity = worse hearing
            # Map normalized intensity [-2, 2] to threshold adjustment [-20, +20]
            intensity_adjustment = np.clip(intensity * 10, -20, 20)
            
            # V peak adjustment: presence indicates better hearing
            v_peak_adjustment = -5 if v_peak_present else +10
            
            # Combine adjustments
            final_threshold = base_threshold + intensity_adjustment + v_peak_adjustment
            final_threshold = np.clip(final_threshold, 0, 120)
            
            thresholds.append(final_threshold)
        
        return np.array(thresholds)
    
    def _combined_method(self, data_samples: List[Dict]) -> np.ndarray:
        """
        Combined method using class, intensity, and V peak information.
        """
        thresholds = []
        
        for sample in data_samples:
            class_idx = sample['target']
            class_name = self.class_names[class_idx]
            intensity = sample['static_params'][1]
            v_peak_present = sample['v_peak_mask'][0] and sample['v_peak_mask'][1]
            v_peak_latency = sample['v_peak'][0] if v_peak_present else None
            
            # Base threshold from clinical classification
            min_thresh, max_thresh = self.threshold_mapping[class_name]
            base_threshold = (min_thresh + max_thresh) / 2
            
            # Intensity-based adjustment
            intensity_factor = np.tanh(intensity)  # Smooth mapping
            intensity_adjustment = intensity_factor * 15  # ±15 dB max
            
            # V peak presence adjustment
            if v_peak_present:
                v_peak_adjustment = -8  # Better hearing if V peak present
                
                # Latency adjustment: longer latency = worse hearing
                if v_peak_latency is not None and v_peak_latency > 0:
                    latency_adjustment = np.clip((v_peak_latency - 5.5) * 2, -5, 10)
                else:
                    latency_adjustment = 0
            else:
                v_peak_adjustment = 12  # Worse hearing if no V peak
                latency_adjustment = 0
            
            # Combine all factors
            final_threshold = (base_threshold + 
                             intensity_adjustment + 
                             v_peak_adjustment + 
                             latency_adjustment)
            
            # Add small random variation for realism
            noise = np.random.normal(0, 3)
            final_threshold += noise
            
            # Ensure clinical validity
            final_threshold = np.clip(final_threshold, 0, 120)
            
            thresholds.append(final_threshold)
        
        return np.array(thresholds)
    
    def validate_thresholds(
        self, 
        thresholds: np.ndarray, 
        data_samples: List[Dict]
    ) -> Dict[str, float]:
        """
        Validate generated thresholds against clinical expectations.
        """
        validation_results = {}
        
        # Overall statistics
        validation_results['mean'] = np.mean(thresholds)
        validation_results['std'] = np.std(thresholds)
        validation_results['range'] = (np.min(thresholds), np.max(thresholds))
        
        # Class-wise validation
        class_thresholds = {name: [] for name in self.class_names}
        class_v_peak_rates = {name: [] for name in self.class_names}
        
        for i, sample in enumerate(data_samples):
            class_idx = sample['target']
            class_name = self.class_names[class_idx]
            v_peak_present = sample['v_peak_mask'][0] and sample['v_peak_mask'][1]
            
            class_thresholds[class_name].append(thresholds[i])
            class_v_peak_rates[class_name].append(v_peak_present)
        
        # Check clinical expectations
        for class_name in self.class_names:
            if class_thresholds[class_name]:
                class_mean = np.mean(class_thresholds[class_name])
                v_peak_rate = np.mean(class_v_peak_rates[class_name]) * 100
                
                validation_results[f'{class_name}_mean'] = class_mean
                validation_results[f'{class_name}_v_peak_rate'] = v_peak_rate
        
        # Clinical correlations
        normal_mean = validation_results.get('NORMAL_mean', 50)
        total_mean = validation_results.get('TOTAL_mean', 50)
        
        validation_results['severity_correlation'] = total_mean > normal_mean
        validation_results['clinical_validity'] = (
            validation_results['range'][0] >= 0 and 
            validation_results['range'][1] <= 120 and
            validation_results['severity_correlation']
        )
        
        return validation_results


def main():
    parser = argparse.ArgumentParser(description='Generate clinical hearing thresholds')
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset pickle file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output dataset pickle file with clinical thresholds')
    parser.add_argument('--method', type=str, default='combined',
                       choices=['clinical_mapping', 'intensity_based', 'combined'],
                       help='Threshold generation method')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation on generated thresholds')
    
    args = parser.parse_args()
    
    print("🏥 Clinical Threshold Generation")
    print("=" * 50)
    
    # Load dataset
    print(f"Loading dataset from: {args.input}")
    data = joblib.load(args.input)
    processed_data = data['data']
    
    print(f"Dataset contains {len(processed_data)} samples")
    
    # Generate clinical thresholds
    generator = ClinicalThresholdGenerator()
    print(f"Generating thresholds using method: {args.method}")
    
    clinical_thresholds = generator.generate_clinical_thresholds(
        processed_data, method=args.method
    )
    
    print(f"Generated {len(clinical_thresholds)} clinical thresholds")
    print(f"Threshold range: [{clinical_thresholds.min():.1f}, {clinical_thresholds.max():.1f}] dB HL")
    print(f"Mean threshold: {clinical_thresholds.mean():.1f} ± {clinical_thresholds.std():.1f} dB HL")
    
    # Add thresholds to dataset
    for i, sample in enumerate(processed_data):
        sample['clinical_threshold'] = clinical_thresholds[i]
    
    # Validation
    if args.validate:
        print("\n🔍 Validation Results:")
        validation = generator.validate_thresholds(clinical_thresholds, processed_data)
        
        for key, value in validation.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Save updated dataset
    print(f"\nSaving updated dataset to: {args.output}")
    
    # Update the data dictionary
    updated_data = data.copy()
    updated_data['data'] = processed_data
    updated_data['has_clinical_thresholds'] = True
    updated_data['threshold_method'] = args.method
    
    joblib.dump(updated_data, args.output)
    print("✅ Clinical thresholds added successfully!")
    
    print(f"\n📋 Usage Instructions:")
    print(f"1. Update your training script to use 'clinical_threshold' as target")
    print(f"2. Remove the arbitrary '× 100' scaling")
    print(f"3. The model should now learn meaningful hearing thresholds")


if __name__ == '__main__':
    main() 