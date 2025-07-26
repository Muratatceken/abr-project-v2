#!/usr/bin/env python3
"""
Joint Generation Example for ABR Hierarchical U-Net

This script demonstrates how to use the enhanced ABR model for joint generation
of both ABR signals and their corresponding static parameters (age, intensity, 
stimulus_rate, fmp).

Features demonstrated:
- Conditional generation (given static parameters)
- Joint generation (generate both signals and parameters)
- Unconditional generation (generate everything)
- Clinical constraint enforcement
- Uncertainty sampling

Author: AI Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import yaml
import os

# Import the enhanced model
from models.hierarchical_unet import ProfessionalHierarchicalUNet

class JointGenerationDemo:
    """Demonstration class for joint generation capabilities."""
    
    def __init__(self, config_path: str = "training/config_production_improved.yaml"):
        """
        Initialize the joint generation demo.
        
        Args:
            config_path: Path to the training configuration file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model with joint generation enabled
        self.model = self._create_model()
        self.model.eval()
        
        print("Joint generation demo initialized successfully!")
        print(f"Model supports joint generation: {self.model.enable_joint_generation}")
    
    def _create_model(self) -> ProfessionalHierarchicalUNet:
        """Create the enhanced model with joint generation capabilities."""
        model_config = self.config.get('model', {})
        
        model = ProfessionalHierarchicalUNet(
            input_channels=1,
            static_dim=4,  # age, intensity, stimulus_rate, fmp
            base_channels=model_config.get('base_channels', 64),
            n_levels=model_config.get('n_levels', 4),
            sequence_length=200,
            signal_length=200,
            num_classes=model_config.get('n_classes', 5),
            dropout=model_config.get('dropout', 0.15),
            use_attention_heads=model_config.get('use_attention', True),
            predict_uncertainty=model_config.get('use_uncertainty', True),
            enable_joint_generation=model_config.get('enable_joint_generation', True),
            static_param_ranges=model_config.get('static_param_ranges')
        ).to(self.device)
        
        return model
    
    def demonstrate_conditional_generation(
        self, 
        batch_size: int = 4,
        save_plots: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Demonstrate conditional generation: generate ABR signals given static parameters.
        
        Args:
            batch_size: Number of samples to generate
            save_plots: Whether to save visualization plots
            
        Returns:
            Generated outputs dictionary
        """
        print("\n" + "="*50)
        print("CONDITIONAL GENERATION DEMO")
        print("="*50)
        
        # Create some example static parameters
        static_params = self._create_example_static_params(batch_size)
        
        print("Input static parameters:")
        param_names = ['Age', 'Intensity', 'Stimulus Rate', 'FMP']
        for i, (batch_idx, params) in enumerate(static_params.cpu().numpy()):
            if i < 2:  # Show first 2 samples
                print(f"Sample {i+1}: " + ", ".join([f"{name}: {param:.3f}" 
                                                   for name, param in zip(param_names, params)]))
        
        # Generate signals conditionally
        with torch.no_grad():
            outputs = self.model.generate_conditional(
                static_params=static_params,
                noise_level=1.0
            )
        
        print(f"\nGenerated outputs:")
        print(f"- Signal shape: {outputs['recon'].shape}")
        print(f"- Peak predictions: {len(outputs['peak'])} components")
        print(f"- Classification shape: {outputs['class'].shape}")
        print(f"- Threshold shape: {outputs['threshold'].shape}")
        
        if save_plots:
            self._plot_conditional_results(static_params, outputs, "conditional_generation.png")
        
        return outputs
    
    def demonstrate_joint_generation(
        self, 
        batch_size: int = 4,
        temperature: float = 1.0,
        use_constraints: bool = True,
        save_plots: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Demonstrate joint generation: generate both ABR signals and static parameters.
        
        Args:
            batch_size: Number of samples to generate
            temperature: Temperature for parameter sampling (higher = more random)
            use_constraints: Apply clinical constraints
            save_plots: Whether to save visualization plots
            
        Returns:
            Generated outputs dictionary
        """
        print("\n" + "="*50)
        print("JOINT GENERATION DEMO")
        print("="*50)
        
        # Generate both signals and parameters jointly
        with torch.no_grad():
            outputs = self.model.generate_joint(
                batch_size=batch_size,
                device=self.device,
                temperature=temperature,
                use_constraints=use_constraints
            )
        
        print(f"Generated outputs (joint):")
        print(f"- Signal shape: {outputs['recon'].shape}")
        print(f"- Static parameters shape: {outputs['static_params'].shape}")
        print(f"- Peak predictions: {len(outputs['peak'])} components")
        print(f"- Classification shape: {outputs['class'].shape}")
        print(f"- Threshold shape: {outputs['threshold'].shape}")
        
        # Show generated static parameters
        if 'static_params_sampled' in outputs:
            generated_params = outputs['static_params_sampled']
        else:
            generated_params = outputs['static_params']
            if generated_params.dim() == 3:  # With uncertainty
                generated_params = generated_params[:, :, 0]  # Take means
        
        print(f"\nGenerated static parameters:")
        param_names = ['Age', 'Intensity', 'Stimulus Rate', 'FMP']
        for i, params in enumerate(generated_params.cpu().numpy()):
            if i < 2:  # Show first 2 samples
                print(f"Sample {i+1}: " + ", ".join([f"{name}: {param:.3f}" 
                                                   for name, param in zip(param_names, params)]))
        
        if save_plots:
            self._plot_joint_results(outputs, "joint_generation.png")
        
        return outputs
    
    def demonstrate_unconditional_generation(
        self, 
        batch_size: int = 4,
        save_plots: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Demonstrate unconditional generation: generate everything from noise.
        
        Args:
            batch_size: Number of samples to generate
            save_plots: Whether to save visualization plots
            
        Returns:
            Generated outputs dictionary
        """
        print("\n" + "="*50)
        print("UNCONDITIONAL GENERATION DEMO")
        print("="*50)
        
        # Generate everything unconditionally
        with torch.no_grad():
            outputs = self.model.generate_unconditional(
                batch_size=batch_size,
                device=self.device
            )
        
        print(f"Generated outputs (unconditional):")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"- {key} shape: {value.shape}")
            elif isinstance(value, tuple):
                print(f"- {key}: tuple with {len(value)} components")
        
        if save_plots:
            self._plot_unconditional_results(outputs, "unconditional_generation.png")
        
        return outputs
    
    def compare_generation_modes(
        self, 
        batch_size: int = 2,
        save_plots: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compare all three generation modes side by side.
        
        Args:
            batch_size: Number of samples to generate for each mode
            save_plots: Whether to save comparison plots
            
        Returns:
            Dictionary with outputs from all generation modes
        """
        print("\n" + "="*50)
        print("GENERATION MODE COMPARISON")
        print("="*50)
        
        results = {}
        
        # 1. Conditional generation
        static_params = self._create_example_static_params(batch_size)
        with torch.no_grad():
            results['conditional'] = self.model.generate_conditional(static_params)
        
        # 2. Joint generation  
        with torch.no_grad():
            results['joint'] = self.model.generate_joint(batch_size, self.device)
        
        # 3. Unconditional generation
        with torch.no_grad():
            results['unconditional'] = self.model.generate_unconditional(batch_size, self.device)
        
        print("Comparison completed!")
        print(f"Generated {batch_size} samples for each of 3 modes")
        
        if save_plots:
            self._plot_mode_comparison(results, static_params, "mode_comparison.png")
        
        return results
    
    def _create_example_static_params(self, batch_size: int) -> torch.Tensor:
        """Create realistic example static parameters."""
        # Use the parameter ranges from the model
        ranges = self.model.static_param_head.parameter_ranges if hasattr(self.model, 'static_param_head') else {
            'age': (-0.36, 11.37),
            'intensity': (-2.61, 1.99),
            'stimulus_rate': (-6.79, 5.10),
            'fmp': (-0.20, 129.11)
        }
        
        static_params = []
        for _ in range(batch_size):
            params = []
            for param_name in ['age', 'intensity', 'stimulus_rate', 'fmp']:
                min_val, max_val = ranges[param_name]
                # Generate values biased toward center of range
                param_val = np.random.normal(
                    loc=(min_val + max_val) / 2,
                    scale=(max_val - min_val) / 6
                )
                param_val = np.clip(param_val, min_val, max_val)
                params.append(param_val)
            static_params.append(params)
        
        return torch.tensor(static_params, dtype=torch.float32, device=self.device)
    
    def _plot_conditional_results(self, static_params: torch.Tensor, outputs: Dict, filename: str):
        """Plot results from conditional generation."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Conditional Generation Results', fontsize=16)
        
        signals = outputs['recon'].cpu().numpy()
        
        # Plot first 2 signals
        for i in range(min(2, signals.shape[0])):
            ax = axes[i // 2, i % 2]
            ax.plot(signals[i], 'b-', linewidth=1.5)
            ax.set_title(f'Generated ABR Signal {i+1}')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
            # Add parameter info as text
            params = static_params[i].cpu().numpy()
            param_text = f'Age: {params[0]:.2f}, Int: {params[1]:.2f}\nRate: {params[2]:.2f}, FMP: {params[3]:.2f}'
            ax.text(0.05, 0.95, param_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots
        for i in range(2, 4):
            axes[i // 2, i % 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Conditional generation plot saved as {filename}")
    
    def _plot_joint_results(self, outputs: Dict, filename: str):
        """Plot results from joint generation."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Joint Generation Results (Signals + Parameters)', fontsize=16)
        
        signals = outputs['recon'].cpu().numpy()
        
        # Get static parameters
        if 'static_params_sampled' in outputs:
            static_params = outputs['static_params_sampled']
        else:
            static_params = outputs['static_params']
            if static_params.dim() == 3:  # With uncertainty
                static_params = static_params[:, :, 0]  # Take means
        static_params = static_params.cpu().numpy()
        
        # Plot first 2 signals with their generated parameters
        for i in range(min(2, signals.shape[0])):
            ax = axes[i // 2, i % 2]
            ax.plot(signals[i], 'r-', linewidth=1.5, label='Generated Signal')
            ax.set_title(f'Joint Generated ABR Signal {i+1}')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
            # Add generated parameter info
            params = static_params[i]
            param_text = f'Generated Parameters:\nAge: {params[0]:.2f}, Int: {params[1]:.2f}\nRate: {params[2]:.2f}, FMP: {params[3]:.2f}'
            ax.text(0.05, 0.95, param_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
        
        # Hide unused subplots
        for i in range(2, 4):
            axes[i // 2, i % 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Joint generation plot saved as {filename}")
    
    def _plot_unconditional_results(self, outputs: Dict, filename: str):
        """Plot results from unconditional generation."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Unconditional Generation Results', fontsize=16)
        
        signals = outputs['recon'].cpu().numpy()
        
        # Plot first 2 signals
        for i in range(min(2, signals.shape[0])):
            ax = axes[i // 2, i % 2]
            ax.plot(signals[i], 'g-', linewidth=1.5)
            ax.set_title(f'Unconditional ABR Signal {i+1}')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(2, 4):
            axes[i // 2, i % 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Unconditional generation plot saved as {filename}")
    
    def _plot_mode_comparison(self, results: Dict, static_params: torch.Tensor, filename: str):
        """Plot comparison of all generation modes."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Generation Mode Comparison', fontsize=16)
        
        modes = ['conditional', 'joint', 'unconditional']
        colors = ['blue', 'red', 'green']
        
        for mode_idx, mode in enumerate(modes):
            signals = results[mode]['recon'].cpu().numpy()
            
            # Plot first 2 signals for each mode
            for i in range(min(2, signals.shape[0])):
                ax = axes[mode_idx, i]
                ax.plot(signals[i], color=colors[mode_idx], linewidth=1.5)
                ax.set_title(f'{mode.capitalize()} Generation - Sample {i+1}')
                ax.set_xlabel('Time (samples)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                
                # Add mode-specific information
                if mode == 'conditional':
                    params = static_params[i].cpu().numpy()
                    param_text = f'Given: Age:{params[0]:.2f}, Int:{params[1]:.2f}'
                elif mode == 'joint':
                    if 'static_params_sampled' in results[mode]:
                        gen_params = results[mode]['static_params_sampled'][i].cpu().numpy()
                    else:
                        gen_params = results[mode]['static_params'][i]
                        if gen_params.dim() > 0 and len(gen_params) > 2:
                            gen_params = gen_params[0].cpu().numpy()
                        else:
                            gen_params = gen_params.cpu().numpy()
                    param_text = f'Generated: Age:{gen_params[0]:.2f}, Int:{gen_params[1]:.2f}'
                else:  # unconditional
                    param_text = 'No parameters'
                
                ax.text(0.05, 0.95, param_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Mode comparison plot saved as {filename}")


def main():
    """Main demonstration function."""
    print("ABR Joint Generation Demonstration")
    print("=" * 50)
    
    # Create demo instance
    demo = JointGenerationDemo()
    
    # Create output directory
    os.makedirs("outputs/joint_generation_demo", exist_ok=True)
    os.chdir("outputs/joint_generation_demo")
    
    try:
        # Demonstrate different generation modes
        print("\nRunning demonstrations...")
        
        # 1. Conditional generation
        conditional_outputs = demo.demonstrate_conditional_generation(batch_size=4)
        
        # 2. Joint generation
        joint_outputs = demo.demonstrate_joint_generation(batch_size=4, temperature=1.0)
        
        # 3. Unconditional generation
        unconditional_outputs = demo.demonstrate_unconditional_generation(batch_size=4)
        
        # 4. Compare all modes
        comparison_results = demo.compare_generation_modes(batch_size=2)
        
        print("\n" + "="*50)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Generated files:")
        print("- conditional_generation.png")
        print("- joint_generation.png") 
        print("- unconditional_generation.png")
        print("- mode_comparison.png")
        
        print(f"\nModel capabilities:")
        model_info = demo.model.get_model_info()
        for mode, description in model_info['generation_modes'].items():
            print(f"- {mode}: {description}")
            
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 