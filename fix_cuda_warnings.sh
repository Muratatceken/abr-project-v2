#!/bin/bash

# Fix CUDA library registration warnings
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

# Suppress TensorFlow warnings
export PYTHONWARNINGS="ignore"

# Run training with suppressed warnings
echo "🚀 Starting training with CUDA warnings suppressed..."
python train.py "$@"