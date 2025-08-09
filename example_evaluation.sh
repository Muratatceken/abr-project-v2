#!/bin/bash

# Example Evaluation Script for ABR Signal Generation Model
# This script demonstrates how to use the comprehensive evaluation pipeline

echo "🚀 ABR Signal Generation Model - Evaluation Examples"
echo "=================================================="

# Basic evaluation (fast)
echo ""
echo "📊 Example 1: Basic Evaluation"
echo "python evaluate_model.py \\"
echo "    --config configs/config_colab_a100.yaml \\"
echo "    --checkpoint checkpoints/pro/best.pth \\"
echo "    --output_dir evaluation_results_basic"

# Comprehensive evaluation (detailed analysis)
echo ""
echo "🔬 Example 2: Comprehensive Evaluation with Detailed Analysis"
echo "python evaluate_model.py \\"
echo "    --config configs/config_colab_a100.yaml \\"
echo "    --checkpoint checkpoints/pro/best.pth \\"
echo "    --output_dir evaluation_results_detailed \\"
echo "    --detailed_analysis \\"
echo "    --num_samples 500 \\"
echo "    --num_ddim_steps 100"

# Quick test with fewer samples
echo ""
echo "⚡ Example 3: Quick Test (Development)"
echo "python evaluate_model.py \\"
echo "    --config configs/config_colab_a100.yaml \\"
echo "    --checkpoint checkpoints/pro/best.pth \\"
echo "    --output_dir evaluation_results_quick \\"
echo "    --num_samples 50 \\"
echo "    --batch_size 16"

# Custom generation parameters
echo ""
echo "🎯 Example 4: Custom Generation Parameters"
echo "python evaluate_model.py \\"
echo "    --config configs/config_colab_a100.yaml \\"
echo "    --checkpoint checkpoints/pro/best.pth \\"
echo "    --output_dir evaluation_results_custom \\"
echo "    --num_ddim_steps 25 \\"
echo "    --cfg_scale 1.5 \\"
echo "    --device cuda"

# Test the evaluation pipeline first
echo ""
echo "🧪 Example 5: Test Evaluation Pipeline (No Model Required)"
echo "python test_evaluation.py"

echo ""
echo "📚 For more options, run: python evaluate_model.py --help"
echo ""
echo "💡 Note: Make sure you have a trained model checkpoint before running evaluation!"
echo "💡 Train a model first using: python train.py --config configs/config_colab_a100.yaml"