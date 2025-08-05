#!/usr/bin/env python3
"""
Enhanced Evaluation Script for ABR Diffusion Model

This script runs comprehensive evaluation using the enhanced evaluation pipeline.
It provides detailed analysis, visualizations, and recommendations for model improvement.
"""

import torch
import logging
from pathlib import Path
import argparse
from omegaconf import DictConfig, OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """Main evaluation function."""
    
    # Load configuration
    try:
        config = OmegaConf.load(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        if args.model_type == "simplified":
            from models.simplified_abr_model import SimplifiedABRModel
            model = SimplifiedABRModel(
                input_channels=config.data.get('input_channels', 1),
                static_dim=config.data.static_dim,
                signal_length=config.data.signal_length,
                num_classes=config.data.n_classes,
                base_channels=config.model.architecture.get('base_channels', 64),
                n_levels=config.model.architecture.get('n_levels', 3),
                dropout=config.model.architecture.get('dropout', 0.15),
                predict_uncertainty=config.model.architecture.get('predict_uncertainty', True)
            )
        else:
            from models.hierarchical_unet import OptimizedHierarchicalUNet
            model = OptimizedHierarchicalUNet(
                signal_length=config.data.signal_length,
                static_dim=config.data.static_dim,
                n_classes=config.data.n_classes,
                **config.model.architecture
            )
        
        model = model.to(device)
        logger.info(f"Model loaded: {model.__class__.__name__}")
        
        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded checkpoint: {checkpoint_path}")
            else:
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load data
    try:
        from data.dataset import ABRDataModule
        
        data_module = ABRDataModule(
            data_path=config.data.dataset_path,
            batch_size=config.data.dataloader.batch_size,
            num_workers=config.data.dataloader.get('num_workers', 2),
            **config.data.splits
        )
        data_module.setup()
        
        test_loader = data_module.test_dataloader()
        logger.info(f"Test dataset loaded: {len(test_loader.dataset)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Create enhanced evaluator
    try:
        from evaluation.enhanced_evaluator import create_enhanced_evaluator
        
        evaluator = create_enhanced_evaluator(
            model=model,
            config=config,
            device=device,
            output_dir=args.output_dir
        )
        
        logger.info("Enhanced evaluator created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create evaluator: {e}")
        return
    
    # Run comprehensive evaluation
    try:
        logger.info("Starting comprehensive evaluation...")
        
        results = evaluator.evaluate_comprehensive(
            test_loader=test_loader,
            max_samples=args.max_samples,
            save_results=True
        )
        
        # Print summary results
        print("\n" + "="*60)
        print("ENHANCED EVALUATION RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nModel: {results.model_name}")
        print(f"Samples Evaluated: {results.total_samples}")
        print(f"Evaluation Time: {results.evaluation_time:.2f} seconds")
        
        print(f"\n📊 SIGNAL QUALITY:")
        for metric, value in results.signal_metrics.items():
            print(f"  • {metric.upper()}: {value:.4f}")
        
        print(f"\n🎯 CLASSIFICATION:")
        for metric, value in results.classification_metrics.items():
            print(f"  • {metric.upper()}: {value:.4f}")
        
        print(f"\n📈 PEAK DETECTION:")
        for metric, value in results.peak_metrics.items():
            print(f"  • {metric.upper()}: {value:.4f}")
        
        print(f"\n🎚️ THRESHOLD REGRESSION:")
        for metric, value in results.threshold_metrics.items():
            print(f"  • {metric.upper()}: {value:.4f}")
        
        print(f"\n🏥 CLINICAL ASSESSMENT:")
        print(f"  • Diagnostic Value: {results.diagnostic_value:.4f}")
        print(f"  • False Positive Rate: {results.false_positive_rate:.4f}")
        print(f"  • False Negative Rate: {results.false_negative_rate:.4f}")
        
        print(f"\n🌊 DIFFUSION QUALITY:")
        print(f"  • Generation Quality: {results.diffusion_quality:.4f}")
        print(f"  • Generation Diversity: {results.generation_diversity:.4f}")
        print(f"  • Conditional Consistency: {results.conditional_consistency:.4f}")
        
        print(f"\n⚠️ FAILURE CASES: {len(results.failure_cases)}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results.recommendations, 1):
            print(f"  {i}. {rec[:100]}..." if len(rec) > 100 else f"  {i}. {rec}")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("Enhanced evaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ABR Model Evaluation")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_diffusion_fixed.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["hierarchical", "simplified"],
        default="hierarchical",
        help="Type of model to evaluate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/enhanced_evaluation",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to evaluate (for quick testing)"
    )
    
    args = parser.parse_args()
    
    main(args)