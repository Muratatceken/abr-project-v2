# HPO Usage Guide for ABR Transformer

## Table of Contents

1. [Introduction and Overview](#introduction-and-overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Configuration System](#configuration-system)
4. [HPO Modes and Strategies](#hpo-modes-and-strategies)
5. [Search Space Design](#search-space-design)
6. [Execution and Monitoring](#execution-and-monitoring)
7. [Cross-Validation and Evaluation](#cross-validation-and-evaluation)
8. [Results Analysis and Interpretation](#results-analysis-and-interpretation)
9. [Advanced Features](#advanced-features)
10. [Best Practices and Tips](#best-practices-and-tips)
11. [Troubleshooting](#troubleshooting)
12. [Integration with Training Pipeline](#integration-with-training-pipeline)
13. [Examples and Case Studies](#examples-and-case-studies)
14. [API Reference](#api-reference)
15. [Appendices](#appendices)

---

## 1. Introduction and Overview

### What is Hyperparameter Optimization?

Hyperparameter optimization (HPO) is the process of automatically finding the best configuration of hyperparameters for a machine learning model. For ABR (Auditory Brainstem Response) transformers, this includes optimizing parameters like learning rates, model architecture dimensions, training strategies, and data augmentation settings.

### Why HPO Matters for ABR Transformers

ABR signal processing requires careful tuning due to:

- **Signal complexity**: ABR signals have subtle temporal patterns requiring precise model configuration
- **Multi-task nature**: Models must balance signal generation and peak classification
- **Patient variability**: Cross-validation strategies must account for patient-level differences
- **Resource constraints**: Training is computationally expensive, making efficient optimization critical

### Framework Overview

Our HPO system uses **Optuna** with:

- **Tree-structured Parzen Estimator (TPE)** for efficient search
- **Multi-objective optimization** for accuracy vs. efficiency tradeoffs
- **Patient-level cross-validation** to prevent data leakage
- **Multi-task evaluation** for comprehensive model assessment
- **Automated pruning** to terminate unpromising trials early

### Key Benefits

- **Automated optimization**: Reduces manual hyperparameter tuning
- **Robust evaluation**: Patient-level CV ensures generalization
- **Multi-task support**: Optimizes for both signal generation and peak detection
- **Resource efficiency**: Pruning and time limits prevent waste
- **Comprehensive analysis**: Detailed results and visualization

### Expected Outcomes

Properly executed HPO typically yields:

- **5-15% performance improvements** over default parameters
- **Faster convergence** through optimized learning rates and schedules
- **Better generalization** via cross-validation and regularization tuning
- **Balanced multi-task performance** through loss weight optimization

---

## 2. Quick Start Guide

### Prerequisites

Before starting HPO, ensure you have:

**System Requirements:**
- Python 3.8+ with PyTorch, Optuna, and dependencies
- CUDA-capable GPU with 8GB+ VRAM (recommended)
- At least 16GB system RAM
- 20GB+ free disk space for basic optimization
- 50GB+ free disk space for comprehensive optimization

**Data Requirements:**
- Preprocessed ABR dataset in PKL format
- Patient stratification information for CV
- Clinical threshold data for evaluation

**Configuration Files:**
- `configs/train_hpo_optimized.yaml` - HPO-optimized training config
- `configs/hpo_search_space.yaml` - Comprehensive parameter search space

### Environment Setup

```bash
# 1. Navigate to project directory
cd /path/to/abr-project-v2

# 2. Activate Python environment
conda activate abr_env  # or your environment name

# 3. Verify installation
python -c "import torch, optuna, yaml; print('Dependencies OK')"

# 4. Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 5. Validate configurations
python scripts/train_with_hpo.py --mode create_search_space --search_space_path configs/hpo_search_space.yaml
```

### Your First HPO Run

**Option 1: Using the convenience script (recommended)**

```bash
# Quick optimization with core parameters (30 trials, ~10 hours)
./scripts/run_hpo_optimized.sh basic

# Check progress and results
./scripts/run_hpo_optimized.sh analysis
```

**Option 2: Direct command-line usage**

```bash
# Basic HPO with manual configuration
python scripts/train_with_hpo.py \
    --config configs/train_hpo_optimized.yaml \
    --mode hpo \
    --n_trials 30 \
    --timeout 1200 \
    --save_dir my_hpo_results \
    --cv_folds 3
```

### Understanding Initial Results

After your first run, check:

1. **Results directory**: `hpo_results_*/TIMESTAMP/`
2. **Main results file**: `hpo_results.json`
3. **Analysis plots**: `hpo_analysis_plots.png`
4. **Log files**: `*.log` for detailed execution logs

**Key metrics to examine:**
- **Best objective value**: Lower is better for loss-based metrics
- **Parameter importance**: Which parameters matter most
- **Optimization history**: Convergence and trial efficiency
- **Cross-validation scores**: Consistency across folds

### Common First-Time Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch_size in search space or use `--cv_folds 3`

**Issue**: "Config not found" error
**Solution**: Run from project root directory with correct paths

**Issue**: No improvement over many trials  
**Solution**: Check search space ranges and increase diversity

**Issue**: Trials failing immediately
**Solution**: Validate base config with single training run first

---

## 3. Configuration System

### Configuration Hierarchy

The HPO system uses a hierarchical configuration approach:

1. **Base training config** (`configs/train.yaml`)
2. **HPO-optimized config** (`configs/train_hpo_optimized.yaml`)  
3. **Search space definition** (`configs/hpo_search_space.yaml`)
4. **Runtime parameter overrides** (command-line arguments)

### HPO-Optimized Configuration

`configs/train_hpo_optimized.yaml` is specifically tuned for efficient HPO:

**Key optimizations:**
- **Reduced max_epochs** (30 vs 100) for faster trials
- **Aggressive early stopping** (patience: 10 vs 20)
- **Disabled expensive features** (visualization, detailed logging)
- **Cross-validation enabled** with patient-level stratification
- **Multi-objective optimization** configured
- **Resource-efficient settings** (gradient checkpointing, mixed precision)

**HPO-specific sections:**

```yaml
# HPO configuration
hyperparameter_optimization:
  enabled: true
  n_trials: 50
  timeout: 1800  # 30 minutes per trial
  sampler: 'tpe'
  pruner: 'median'
  
  # Multi-objective optimization
  multi_objective:
    enabled: true
    objectives: ['val_combined_score', 'total_training_time']
    directions: ['maximize', 'minimize']

# Cross-validation configuration  
cross_validation:
  enabled: true
  n_folds: 5
  stratified: true
  patient_level: true  # Prevents data leakage
```

### Search Space Configuration

`configs/hpo_search_space.yaml` defines the parameter search space:

**Parameter categories:**
- **Model architecture**: d_model, n_layers, n_heads, dropout
- **Training parameters**: learning_rate, batch_size, weight_decay
- **Loss configuration**: stft_weight, focal_loss parameters
- **Multi-task learning**: task weights and strategies
- **Augmentation**: mixup, cutmix, noise parameters
- **Advanced features**: curriculum learning, scheduling

**Parameter types:**

```yaml
# Categorical parameters
d_model:
  type: 'categorical'
  choices: [128, 256, 384, 512]

# Float parameters with log scaling
learning_rate:
  type: 'float'
  low: 1e-5
  high: 1e-2
  log: true

# Integer parameters with step size
n_layers:
  type: 'int'
  low: 4
  high: 12
  step: 2
```

### Parameter Constraints

**Divisibility constraints:**
- `n_heads` must divide `d_model` evenly
- Automatically enforced during optimization

**Conditional parameters:**
- Focal loss parameters only used when `use_focal_loss: true`
- Curriculum parameters only used when `use_curriculum: true`

**Custom constraints:**
- Amplitude scaling ranges validated for sensible bounds
- Multi-task weights normalized to sum to reasonable values

### Customizing Configurations

**To modify search ranges:**

1. Edit `configs/hpo_search_space.yaml`
2. Adjust parameter ranges based on domain knowledge
3. Add new parameters by following existing patterns

**To change optimization settings:**

1. Edit `configs/train_hpo_optimized.yaml`
2. Modify `hyperparameter_optimization` section
3. Adjust resource limits and evaluation settings

**To add custom parameters:**

1. Define in search space YAML
2. Add mapping in `optimization/hyperparameter_optimization.py`
3. Ensure target config path exists

---

## 4. HPO Modes and Strategies

### Basic HPO Mode

**Purpose**: Quick optimization focusing on core parameters for fast results

**Configuration**:
- **30 trials** with 20-minute timeout per trial
- **3-fold cross-validation** for faster evaluation
- **Core parameters**: learning_rate, d_model, batch_size, dropout
- **Total time**: ~10 hours

**When to use**:
- Initial exploration of parameter space
- Limited computational resources
- Need quick baseline optimization

**Command**:
```bash
./scripts/run_hpo_optimized.sh basic [trials] [timeout] [save_dir]
```

### Comprehensive HPO Mode

**Purpose**: Full optimization using complete search space

**Configuration**:
- **100 trials** with 30-minute timeout per trial
- **5-fold patient-level cross-validation**
- **Complete search space** from `hpo_search_space.yaml`
- **Total time**: ~50 hours

**When to use**:
- Production model development
- Maximum performance requirements
- Sufficient computational budget available

**Command**:
```bash
./scripts/run_hpo_optimized.sh full [trials] [timeout] [save_dir]
```

### AutoML Mode

**Purpose**: Automated machine learning with minimal user intervention

**Features**:
- **Simplified AutoML pipeline** to search configurations
- **Advanced features** (ensembles/feature engineering) are roadmap items
- **Automated hyperparameter optimization** strategies
- **Basic self-tuning** capabilities

**Configuration**:
- **50 trials** with adaptive timeout
- **Simplified automated configuration search**
- **Basic optimization strategies**

> **Roadmap**: Advanced AutoML features including automated ensemble methods, feature engineering, and meta-learning are planned for future releases.

**When to use**:
- Minimal expertise in hyperparameter tuning
- Want simplified automated optimization
- Need basic AutoML capabilities with reasonable defaults

**Command**:
```bash
./scripts/run_hpo_optimized.sh automl [trials] [save_dir]
```

### Multi-Objective Optimization

**Purpose**: Balance multiple competing objectives (accuracy vs efficiency)

**Objectives**:
- **Primary**: Maximize validation combined score
- **Secondary**: Minimize total training time
- **Analysis**: Pareto frontier optimization

**Configuration**:
- **75 trials** with 30-minute timeout
- **TPE sampler** with multi-objective support
- **Pareto frontier analysis**

**When to use**:
- Need to balance performance and efficiency
- Resource-constrained environments
- Want multiple optimal solutions

**Command**:
```bash
./scripts/run_hpo_optimized.sh multi_obj [trials] [timeout] [save_dir]
```

### Analysis Mode

**Purpose**: Analyze and visualize existing HPO results

**Features**:
- **Parameter importance** ranking
- **Optimization history** visualization
- **Statistical analysis** of results
- **Performance comparisons**

**Outputs**:
- Analysis plots and charts
- Parameter importance rankings
- Optimization convergence analysis
- Best configuration summaries

**Command**:
```bash
./scripts/run_hpo_optimized.sh analysis [results_dir]
```

### Resume Mode

**Purpose**: Continue interrupted optimization studies

**Features**:
- **Resume from checkpoint** using existing study state
- **Add additional trials** to existing optimization
- **Preserve optimization history**
- **Maintain sampler state**

**When to use**:
- Optimization was interrupted
- Want to extend completed study
- Need more trials for convergence

**Command**:
```bash
./scripts/run_hpo_optimized.sh resume [study_dir] [additional_trials]
```

---

## 5. Search Space Design

### Parameter Categories

#### Model Architecture Parameters

**Core transformer dimensions:**
```yaml
d_model:
  type: 'categorical'
  choices: [128, 256, 384, 512]
  description: "Model dimension - affects capacity and memory"
  
n_layers:
  type: 'int'
  low: 4
  high: 12
  step: 2
  description: "Number of transformer layers"
  
n_heads:
  type: 'categorical'
  choices: [4, 6, 8, 12, 16]
  constraints: "Must divide d_model evenly"
```

**Regularization:**
```yaml
dropout:
  type: 'float'
  low: 0.0
  high: 0.3
  step: 0.05
  description: "Dropout probability for regularization"
```

#### Training Parameters

**Learning optimization:**
```yaml
learning_rate:
  type: 'float'
  low: 1e-5
  high: 1e-2
  log: true
  description: "Initial learning rate (log-uniform sampling)"
  
batch_size:
  type: 'categorical'
  choices: [8, 16, 32, 48, 64]
  description: "Training batch size"
  
weight_decay:
  type: 'float'
  low: 1e-6
  high: 1e-2
  log: true
  description: "L2 regularization weight"
```

#### Loss Configuration

**Multi-domain loss weights:**
```yaml
stft_weight:
  type: 'float'
  low: 0.0
  high: 0.3
  step: 0.02
  description: "STFT loss weight for frequency domain"
  
use_focal_loss:
  type: 'categorical'
  choices: [true, false]
  description: "Use focal loss for class imbalance"
  
focal_alpha:
  type: 'float'
  low: 0.1
  high: 0.9
  step: 0.1
  conditional: "use_focal_loss == true"
```

### Parameter Constraints

#### Architectural Constraints

**Attention mechanism compatibility:**
- `d_model` must be divisible by `n_heads`
- Automatically enforced during optimization
- Invalid combinations are filtered out

**Memory constraints:**
- Large `d_model` × `n_layers` combinations may exceed GPU memory
- Batch size automatically adjusted based on model size

### Designing Effective Search Spaces

#### Best Practices

1. **Start broad, then narrow**: Begin with wide ranges, refine based on results
2. **Use domain knowledge**: Incorporate known good parameter ranges
3. **Consider interactions**: Some parameters have strong dependencies
4. **Balance exploration vs exploitation**: Mix broad and focused ranges

#### Parameter Importance Guidance

**High impact parameters** (optimize first):
- `learning_rate`: Most critical for convergence
- `d_model`: Core capacity parameter
- `batch_size`: Affects both performance and training dynamics
- `peak_classification_weight`: Critical for multi-task balance

**Medium impact parameters**:
- `n_layers`: Model depth vs overfitting tradeoff
- `dropout`: Regularization vs capacity balance
- `weight_decay`: Generalization improvement
- `stft_weight`: Multi-domain loss balance

**Low impact parameters** (fine-tune last):
- `n_heads`: Usually 8 works well for most d_model values
- `augmentation_strength`: Moderate impact on robustness
- `ema_decay`: Fine-tuning parameter

---

## 6. Execution and Monitoring

### Using the Convenience Script

The `scripts/run_hpo_optimized.sh` script provides a user-friendly interface for running different HPO modes:

**Basic execution:**
```bash
# Quick start with defaults
./scripts/run_hpo_optimized.sh basic

# Custom parameters
./scripts/run_hpo_optimized.sh basic 50 1800 my_results
```

**Advanced modes:**
```bash
# Comprehensive optimization
./scripts/run_hpo_optimized.sh full 100 2400 comprehensive_results

# Multi-objective optimization
./scripts/run_hpo_optimized.sh multi_obj 75 1800 multi_obj_results

# Analysis of existing results
./scripts/run_hpo_optimized.sh analysis results_directory/
```

### Monitoring Logs and Progress

**Real-time log monitoring:**
```bash
# Follow the main log file
tail -f hpo_results_*/TIMESTAMP/hpo_*.log

# Monitor both main and resource logs
tail -f hpo_results_*/TIMESTAMP/*.log
```

**Understanding log output:**
- **Trial completion**: Shows objective value and parameters for each trial
- **Cross-validation progress**: CV fold results with mean and std deviation
- **Resource usage**: GPU memory, system memory, and utilization statistics
- **Best parameters**: Updated whenever a new best configuration is found

### Resource Monitoring Flags

**Built-in monitoring options:**
```bash
# Disable GPU monitoring (useful for CPU-only systems)
./scripts/run_hpo_optimized.sh basic --no-gpu-monitor

# Disable memory monitoring (reduces log file size)
./scripts/run_hpo_optimized.sh basic --no-memory-monitor

# Quiet mode (minimal console output)
./scripts/run_hpo_optimized.sh basic --quiet
```

**Custom resource monitoring:**
```bash
# Monitor GPU usage externally
watch -n 30 nvidia-smi

# Monitor system resources
htop  # or top on macOS
```

### Handling Interruptions and Resume

**Graceful shutdown:**
```bash
# Send interrupt signal to allow cleanup
pkill -TERM -f train_with_hpo

# Check for saved results before termination
ls -la hpo_results_*/*/hpo_results.json
```

**Resuming interrupted optimization:**
```bash
# Resume with additional trials
./scripts/run_hpo_optimized.sh resume hpo_results_*/TIMESTAMP 25

# Resume with different settings
./scripts/run_hpo_optimized.sh resume study_dir 50
```

**Recovery from failures:**
- Results are saved incrementally after each trial
- Partial results can be analyzed even from incomplete runs
- Study state is preserved for resuming from the last successful trial

---

## 9. Advanced Features

### Multi-Objective Optimization Setup

**Configuration for multiple objectives:**
```yaml
# In train_hpo_optimized.yaml
hyperparameter_optimization:
  multi_objective:
    enabled: true
    objectives: ['val_combined_score', 'total_training_time']
    directions: ['maximize', 'minimize']
```

**Running multi-objective optimization:**
```bash
# Balanced accuracy vs efficiency
./scripts/run_hpo_optimized.sh multi_obj 75 1800 pareto_optimization
```

**Analyzing Pareto frontier:**
- Multiple optimal solutions trading off between objectives
- Knee point identification for balanced performance
- Dominated vs Pareto-optimal solution analysis

### Advanced Sampler and Pruner Tuning

**Sampler options and when to use them:**
```bash
# TPE (Tree-structured Parzen Estimator) - default, best for most cases
--sampler tpe

# Random sampling - good baseline, parallel-friendly
--sampler random

# CMA-ES - effective for continuous parameters, smaller search spaces
--sampler cmaes
```

**Pruner strategies:**
```bash
# Median pruner - stops trials performing worse than median
--pruner median

# Hyperband - more sophisticated, based on successive halving
--pruner hyperband

# No pruning - let all trials complete (for debugging)
--pruner none
```

**Custom sampler configuration:**
```python
# Advanced TPE settings in code
import optuna
sampler = optuna.samplers.TPESampler(
    n_startup_trials=15,  # Random trials before TPE
    n_ei_candidates=30,   # Expected improvement candidates
    gamma=0.25           # Top gamma fraction for modeling
)
```

### Distributed HPO (Future Roadmap)

**Planned features:**
- Multi-GPU parallel trial execution
- Distributed optimization across multiple nodes
- Shared study storage for team collaboration
- Auto-scaling based on resource availability

**Current limitations:**
- Single-machine execution only
- Sequential trial processing
- Local file-based result storage

---

## 12. Integration with Training Pipeline

### Exporting Best Parameters for Final Training

**Extract best configuration:**
```python
import json

# Load HPO results
with open('hpo_results/TIMESTAMP/hpo_results.json') as f:
    results = json.load(f)

best_params = results['best_params']
print(f"Best learning rate: {best_params['learning_rate']}")
print(f"Best d_model: {best_params['d_model']}")
print(f"Best batch_size: {best_params['batch_size']}")
```

**Create optimized training config:**
```bash
# Use the convenience script to generate optimized config
python3 - <<'PY'
import yaml
import json

# Load HPO results
with open('hpo_results/TIMESTAMP/hpo_results.json') as f:
    hpo_results = json.load(f)

# Load base config
with open('configs/train.yaml') as f:
    config = yaml.safe_load(f)

# Apply best parameters
best_params = hpo_results['best_params']
config['optim']['lr'] = best_params.get('learning_rate', config['optim']['lr'])
config['model']['d_model'] = best_params.get('d_model', config['model']['d_model'])
config['loader']['batch_size'] = best_params.get('batch_size', config['loader']['batch_size'])

# Add other optimized parameters...

# Save optimized config
with open('configs/train_optimized.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Optimized training config saved to: configs/train_optimized.yaml")
PY
```

### Using HPO Results in Production Training

**Full training with optimized parameters:**
```bash
# Train final model with best hyperparameters
python train.py --config configs/train_optimized.yaml --exp_name final_model

# Use longer training schedule for production
python train.py --config configs/train_optimized.yaml \
    --override "trainer.max_epochs: 200, trainer.early_stop_patience: 40"
```

**Checkpoint and model management:**
```bash
# Create production-ready checkpoint
python train.py --config configs/train_optimized.yaml \
    --exp_name production_model \
    --override "trainer.ckpt_dir: checkpoints/production"

# Validate final model
python eval.py --config configs/train_optimized.yaml \
    --checkpoint checkpoints/production/best.pt \
    --save_dir evaluation_results/production
```

### Ensemble Methods with Top Configurations

**Train ensemble from top HPO results:**
```python
import json

# Load HPO results and get top 5 configurations
with open('hpo_results.json') as f:
    results = json.load(f)

# Sort trials by performance
trials = results.get('all_trials', [])
top_configs = sorted(trials, key=lambda x: x['value'], reverse=True)[:5]

# Train ensemble models
for i, trial in enumerate(top_configs):
    config_path = f"configs/ensemble_model_{i+1}.yaml"
    # Generate config for this trial...
    # Train model with this config...
```

**Production deployment considerations:**
- Use cross-validation results to assess generalization
- Validate on holdout test set before deployment
- Consider computational requirements for production
- Implement model versioning and rollback strategies

---

## 7. Cross-Validation and Evaluation

### Patient-Level Stratified Cross-Validation

**Why patient-level CV?**
- **Prevents data leakage**: Ensures no patient appears in both training and validation
- **Realistic evaluation**: Mimics real-world deployment on new patients
- **Robust estimates**: Reduces variance in performance estimates

**Configuration:**
```yaml
cross_validation:
  enabled: true
  n_folds: 5
  stratified: true
  patient_level: true
  cv_type: 'stratified_patient'
```

**Implementation details:**
- Patients are grouped by ID to prevent leakage
- Stratification maintains class balance across folds
- Each fold trains on 80% of patients, validates on 20%

### Multi-Task Evaluation Metrics

**Primary metrics:**
- **Combined score**: Weighted combination of signal and peak performance
- **Signal generation loss**: MSE and STFT loss for waveform quality
- **Peak classification**: F1-score, AUC, precision, recall
- **Cross-validation consistency**: Standard deviation across folds

**Metric computation:**
```python
# Combined score calculation
signal_weight = 0.7
peak_weight = 0.3
combined_score = signal_weight * (1 - signal_loss) + peak_weight * peak_f1
```

### Statistical Validation

**Confidence intervals:**
- 95% CI computed across CV folds
- Helps assess result reliability
- Identifies parameters with consistent performance

**Significance testing:**
- Wilcoxon signed-rank test for parameter comparisons
- Bonferroni correction for multiple comparisons
- Effect size estimation (Cohen's d)

---

## 8. Results Analysis and Interpretation

### Understanding Optuna Study Results

**Key result files:**
- `hpo_results.json`: Main results with best parameters and history
- `hpo_analysis_plots.png`: Visualization plots
- `*.log`: Detailed execution logs

**Result structure:**
```json
{
  "study_name": "abr_transformer_hpo",
  "best_value": 0.823456,
  "best_params": {
    "learning_rate": 0.000324,
    "d_model": 256,
    "batch_size": 32,
    "dropout": 0.15
  },
  "optimization_history": [0.654, 0.678, 0.692, ...],
  "parameter_importance": {
    "learning_rate": 0.342,
    "d_model": 0.234,
    "batch_size": 0.156,
    ...
  }
}
```

### Parameter Importance Analysis

**Interpretation guidelines:**
- **High importance (>0.3)**: Critical parameters requiring careful tuning
- **Medium importance (0.1-0.3)**: Moderate impact, secondary optimization targets
- **Low importance (<0.1)**: Minor impact, can use reasonable defaults

**Common patterns:**
- Learning rate typically most important (0.3-0.5)
- Model architecture parameters usually medium importance (0.1-0.3)
- Augmentation parameters typically low importance (<0.1)

### Optimization History Analysis

**Convergence patterns:**
- **Rapid initial improvement**: Good search space coverage
- **Plateau after 20-30 trials**: May need more exploration
- **Continued improvement after 50+ trials**: Well-designed space

**Warning signs:**
- **No improvement after 10 trials**: Search space too narrow or poor
- **High variance**: Evaluation methodology issues
- **Monotonic decline**: Optimization direction error

### Multi-Objective Analysis

**Pareto frontier interpretation:**
- **Dominated solutions**: Clearly inferior, can be ignored
- **Pareto-optimal solutions**: Trade-offs between objectives
- **Knee point**: Best compromise between objectives

**Selection criteria:**
- **Performance-first**: Choose highest accuracy on frontier
- **Efficiency-first**: Choose fastest training on frontier
- **Balanced**: Choose knee point of Pareto frontier

---

## 9. Best Practices and Tips

### Computational Resource Planning

**Hardware recommendations:**
- **GPU**: RTX 3080/4080 or better with 12GB+ VRAM
- **CPU**: 8+ cores for parallel data loading
- **RAM**: 32GB+ for large batch sizes and CV
- **Storage**: NVMe SSD with 100GB+ free space

**Time estimation:**
- **Basic HPO (30 trials)**: 8-12 hours
- **Full HPO (100 trials)**: 48-72 hours
- **AutoML (50 trials)**: 24-36 hours

**Cost optimization:**
- Use cloud instances with preemptible/spot pricing
- Schedule optimization during off-peak hours
- Use model checkpointing for resumability

### Search Space Design Principles

**Start with proven defaults:**
```yaml
# Known good starting points
learning_rate: [1e-4, 3e-4, 1e-3]  # Narrow around typical values
d_model: [256, 384, 512]            # Skip very small/large models initially
batch_size: [16, 32, 48]            # Memory-feasible range
```

**Progressive refinement:**
1. **Phase 1**: Wide exploration with 30-50 trials
2. **Phase 2**: Narrow around promising regions with 50+ trials
3. **Phase 3**: Fine-tuning with focused ranges

**Parameter interaction awareness:**
- Large models need lower learning rates
- Small batch sizes need higher learning rates
- More layers need more regularization (dropout, weight decay)

### Common Pitfalls and Solutions

**Pitfall: Over-optimization to validation set**
- **Problem**: HPO finds parameters that overfit to CV validation splits
- **Solution**: Use nested CV with holdout test set

**Pitfall: Insufficient trial budget**
- **Problem**: Stopping optimization before convergence
- **Solution**: Monitor convergence, extend trials if still improving

**Pitfall: Poor search space boundaries**
- **Problem**: Optimal values at search space boundaries
- **Solution**: Extend ranges and re-run optimization

**Pitfall: Ignoring computational constraints**
- **Problem**: Optimal parameters too expensive for production
- **Solution**: Include efficiency metrics in multi-objective optimization

### Performance Optimization Tips

**Speed up individual trials:**
- Reduce max_epochs for HPO (30 vs 100 for final training)
- Use 3-fold CV instead of 5-fold during exploration
- Enable mixed precision training (AMP)
- Use gradient checkpointing to allow larger batches

**Speed up overall optimization:**
- Use median pruner to stop unpromising trials
- Start with smaller search space, expand promising regions
- Use warm-start from previous optimization studies

**Improve result quality:**
- Use patient-level stratified CV
- Include multiple evaluation metrics
- Validate on holdout set after HPO completion
- Use ensemble methods with top configurations

---

## 10. Troubleshooting

### Common Error Messages

**"CUDA out of memory"**
- **Cause**: Batch size too large for GPU memory
- **Solutions**: 
  - Reduce batch_size in search space
  - Enable gradient checkpointing
  - Use gradient accumulation
  - Reduce model size (d_model, n_layers)

**"Config file not found"**
- **Cause**: Working directory or path issues
- **Solutions**:
  - Run from project root directory
  - Use absolute paths in command
  - Verify file existence with `ls -la configs/`

**"Invalid YAML syntax"**
- **Cause**: Malformed configuration files
- **Solutions**:
  - Validate YAML: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
  - Check indentation and special characters
  - Use YAML validator online

**"No improvement after many trials"**
- **Cause**: Poor search space or optimization setup
- **Solutions**:
  - Verify base configuration works standalone
  - Check search space ranges are reasonable
  - Examine trial logs for systematic failures
  - Try different sampler (random vs TPE)

### Memory and GPU Issues

**GPU memory management:**
```bash
# Monitor GPU usage
watch -n 5 nvidia-smi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check memory allocation
python -c "import torch; print(torch.cuda.memory_summary())"
```

**System memory issues:**
- Reduce number of data loader workers
- Decrease batch size and gradient accumulation
- Use memory profiling to identify leaks

### Configuration Validation Errors

**Search space validation:**
```python
# Test search space loading
from optimization.hyperparameter_optimization import HyperparameterSpace
space = HyperparameterSpace()
space.load_from_config('configs/hpo_search_space.yaml')
print("Search space loaded successfully")
```

**Training config validation:**
```bash
# Test training configuration
python -c "import yaml; yaml.safe_load(open('configs/train_hpo_optimized.yaml')); print('Training config valid')"
```

### Data Loading and Preprocessing Issues

**Data file validation:**
```python
# Check data file integrity
import pickle
with open('data/processed/ultimate_dataset_with_clinical_thresholds.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"Data loaded: {len(data)} samples")
```

**Cross-validation setup issues:**
- Verify patient IDs are consistent
- Check class balance across folds
- Ensure no data leakage between splits

### Optuna-Specific Problems

**Study storage issues:**
```python
# Clean up corrupted study storage
import optuna
study = optuna.create_study(study_name="new_study", direction="maximize")
```

**Trial pruning too aggressive:**
- Increase pruner patience
- Use less aggressive pruning strategy
- Disable pruning for debugging

**Sampler convergence issues:**
- Increase n_startup_trials for TPE
- Try different samplers (TPE vs Random vs CmaEs)
- Check parameter importance for guidance

---

## 11. Examples and Case Studies

### Complete Worked Example

**Scenario**: Optimize ABR transformer for new clinical dataset

**Step 1: Environment setup**
```bash
cd /path/to/abr-project
conda activate abr_env
# Validate setup
python -c "import yaml; yaml.safe_load(open('configs/train_hpo_optimized.yaml')); print('Config valid')"
```

**Step 2: Initial exploration (Basic HPO)**
```bash
# Quick exploration focusing on core parameters
./scripts/run_hpo_optimized.sh basic 30 1200 initial_exploration

# Expected results: 10-15% improvement over defaults
# Time: ~8 hours on RTX 3080
```

**Step 3: Analysis and refinement**
```bash
# Analyze initial results
./scripts/run_hpo_optimized.sh analysis initial_exploration/TIMESTAMP

# Key findings from example:
# - learning_rate: 3e-4 optimal (importance: 0.45)
# - d_model: 256 best balance of performance/speed (importance: 0.28)
# - batch_size: 32 optimal for this GPU (importance: 0.12)
```

**Step 4: Comprehensive optimization**
```bash
# Full optimization with refined search space
# (edit hpo_search_space.yaml based on initial findings)
./scripts/run_hpo_optimized.sh full 100 1800 comprehensive_hpo
```

**Step 5: Final validation**
```bash
# Train final model with best parameters
python train.py --config best_config.yaml --resume ""

# Validate on holdout test set
python eval.py --config best_config.yaml --checkpoint best_model.pt
```

**Results achieved:**
- **Baseline performance**: 0.756 combined score
- **After basic HPO**: 0.823 combined score (+8.9%)
- **After full HPO**: 0.847 combined score (+12.0%)
- **Time investment**: 60 hours total optimization

### Multi-Objective Case Study

**Scenario**: Balance accuracy with training efficiency for production deployment

**Configuration:**
```bash
# Multi-objective optimization
./scripts/run_hpo_optimized.sh multi_obj 75 1800 production_optimization
```

**Pareto frontier analysis:**
- **High accuracy solution**: 0.851 score, 45 min training time
- **Balanced solution**: 0.832 score, 28 min training time (knee point)
- **Fast solution**: 0.808 score, 18 min training time

**Selection rationale:**
- Production constraints: <30 min training time
- Minimum performance: >0.825 combined score
- **Chosen**: Balanced solution (0.832, 28 min)

### Performance Improvement Case Study

**Original model performance:**
- Combined score: 0.742
- Signal MSE: 0.0234
- Peak F1: 0.798
- Training time: 52 minutes

**HPO-optimized performance:**
- Combined score: 0.856 (+15.4%)
- Signal MSE: 0.0198 (-15.4%)
- Peak F1: 0.847 (+6.1%)
- Training time: 38 minutes (-27%)

**Key optimized parameters:**
- learning_rate: 0.00032 (was 0.0001)
- d_model: 384 (was 256)
- batch_size: 48 (was 32)
- stft_weight: 0.12 (was 0.15)
- peak_classification_weight: 0.65 (was 0.5)

---

## 12. API Reference

### Command-Line Interface

**Main HPO script:**
```bash
python scripts/train_with_hpo.py [OPTIONS]
```

**Key options:**
- `--config PATH`: Base training configuration
- `--search_space_path PATH`: External search space definition
- `--mode {hpo,automl,analyze}`: Operation mode
- `--n_trials INT`: Number of optimization trials
- `--timeout INT`: Timeout per trial in seconds
- `--sampler {tpe,random,cmaes}`: Optimization algorithm
- `--pruner {median,hyperband,none}`: Trial pruning strategy
- `--save_dir PATH`: Results output directory
- `--cv_folds INT`: Cross-validation folds

### Python API

**Core classes:**

```python
from optimization.hyperparameter_optimization import (
    HPOConfig, HyperparameterSpace, HPOObjective, OptunaTuner
)

# Create HPO configuration
config = HPOConfig(
    study_name="my_study",
    n_trials=50,
    direction="maximize"
)

# Define search space
space = HyperparameterSpace()
space.load_from_config("configs/hpo_search_space.yaml")

# Create objective function
objective = HPOObjective(
    train_fn=my_training_function,
    eval_fn=my_evaluation_function,
    hyperparameter_space=space,
    base_config=base_config
)

# Run optimization
tuner = OptunaTuner(config, objective, "results/")
results = tuner.optimize()
```

### Configuration Schema

**HPO configuration schema:**
```yaml
hyperparameter_optimization:
  enabled: boolean
  n_trials: integer (1-1000)
  timeout: integer (seconds)
  sampler: string ("tpe"|"random"|"cmaes")
  pruner: string ("median"|"hyperband"|"none")
  multi_objective:
    enabled: boolean
    objectives: list of strings
    directions: list of strings ("maximize"|"minimize")
```

**Search space parameter schema:**
```yaml
parameter_name:
  type: string ("categorical"|"float"|"int")
  # For categorical
  choices: list of values
  # For float/int
  low: number
  high: number
  step: number (optional)
  log: boolean (optional)
  # Documentation
  description: string
  optimization_hint: string
  conditional: string (optional)
```

---

## 15. Appendices

### A. Hardware Requirements and Recommendations

**Minimum requirements:**
- GPU: GTX 1080 Ti with 8GB VRAM
- CPU: 4 cores, 8 threads
- RAM: 16GB
- Storage: 50GB free space on SSD

**Recommended setup:**
- GPU: RTX 3080/4080 with 12GB+ VRAM
- CPU: 8+ cores with high single-thread performance
- RAM: 32GB DDR4-3200
- Storage: 200GB free on NVMe SSD

**Cloud computing options:**
- AWS: p3.2xlarge or g4dn.2xlarge instances
- Google Cloud: n1-standard-8 with T4 or V100 GPU
- Azure: NC6s_v3 or ND40rs_v2 instances

### B. Performance Benchmarks

**Training time estimates (single trial):**

| Configuration | GPU | Time | Memory |
|--------------|-----|------|---------|
| Basic (d_model=128, n_layers=4) | RTX 3080 | 8 min | 4GB |
| Standard (d_model=256, n_layers=6) | RTX 3080 | 15 min | 6GB |
| Large (d_model=512, n_layers=8) | RTX 3080 | 35 min | 10GB |
| Large (d_model=512, n_layers=8) | RTX 4090 | 22 min | 10GB |

**HPO completion time estimates:**

| Mode | Trials | Expected Time | GPU Hours |
|------|--------|---------------|-----------|
| Basic HPO | 30 | 8-12 hours | 10 |
| Full HPO | 100 | 48-72 hours | 60 |
| AutoML | 50 | 24-36 hours | 30 |
| Multi-objective | 75 | 36-54 hours | 45 |

### C. Troubleshooting Checklist

**Before starting HPO:**
- [ ] GPU has sufficient memory for largest expected model
- [ ] Data files are accessible and valid
- [ ] Base configuration trains successfully
- [ ] Search space YAML loads without errors
- [ ] Sufficient disk space available (20GB+ for basic, 100GB+ for full)

**If trials are failing:**
- [ ] Check individual trial logs for specific errors
- [ ] Verify search space ranges are reasonable
- [ ] Test base configuration with manual hyperparameters
- [ ] Monitor GPU memory usage during trials
- [ ] Check data loading and preprocessing

**If no improvement seen:**
- [ ] Verify optimization direction (maximize vs minimize)
- [ ] Check if search space includes known good values
- [ ] Examine parameter importance to focus search
- [ ] Try different sampler (TPE vs random)
- [ ] Increase trial budget if converging slowly

**For performance issues:**
- [ ] Monitor GPU utilization (should be >80%)
- [ ] Check CPU usage and I/O bottlenecks
- [ ] Verify efficient data loading (prefetch, pin_memory)
- [ ] Consider mixed precision training (AMP)
- [ ] Use gradient checkpointing for memory efficiency

---

This completes the comprehensive HPO Usage Guide. For additional support, consult the project documentation or open an issue on the project repository.

