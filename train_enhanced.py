#!/usr/bin/env python3
"""
Enhanced CLI entrypoint for ABR diffusion training with architectural improvements.

Usage examples:
  python train_enhanced.py --config configs/config_optimized_training.yaml
  python train_enhanced.py --config configs/config_optimized_training.yaml --resume
  python train_enhanced.py --config configs/config_optimized_training.yaml --device cpu
"""

import argparse
import sys
import yaml
from pathlib import Path

from training.enhanced_trainer import EnhancedABRTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced ABR Diffusion Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device: cpu|cuda")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Optional device override
    if args.device is not None:
        cfg.setdefault('training', {})['device'] = args.device

    trainer = EnhancedABRTrainer(cfg)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)