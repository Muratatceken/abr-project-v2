#!/bin/bash

# ABR Transformer Hyperparameter Optimization Script
# 
# This script provides practical examples for running hyperparameter optimization
# on ABR transformer models using different modes and configurations.
#
# Usage: ./scripts/run_hpo_optimized.sh [MODE] [OPTIONS]
#
# Modes:
#   basic       - Quick optimization focusing on core parameters (30 trials)
#   full        - Comprehensive optimization using external search space (100 trials)
#   automl      - Automated machine learning pipeline with ensemble methods
#   multi_obj   - Multi-objective optimization (accuracy vs efficiency)
#   analysis    - Analyze existing HPO results and generate visualizations
#   resume      - Resume interrupted optimization from existing study
#
# Requirements:
#   - Python environment with ABR transformer dependencies
#   - CUDA-capable GPU (recommended)
#   - At least 16GB RAM
#   - configs/train_hpo_optimized.yaml
#   - configs/hpo_search_space.yaml
#
# Author: ABR Transformer Project
# Version: 1.0

set -e  # Exit on any error
set -o pipefail  # Exit on pipeline failures

# =============================================================================
# Configuration Variables
# =============================================================================

# Default configuration
DEFAULT_CONFIG="configs/train_hpo_optimized.yaml"
SEARCH_SPACE_CONFIG="configs/hpo_search_space.yaml"
HPO_SCRIPT="scripts/train_with_hpo.py"

# Default parameters
DEFAULT_TRIALS=30
DEFAULT_TIMEOUT=1800  # 30 minutes per trial
DEFAULT_SAVE_DIR="hpo_results"
DEFAULT_LOG_LEVEL="INFO"

# Resource monitoring
MONITOR_GPU=true
MONITOR_MEMORY=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" >&2
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if file exists and is readable
check_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        error "File not found: $file"
        return 1
    fi
    if [[ ! -r "$file" ]]; then
        error "File not readable: $file"
        return 1
    fi
    return 0
}

# Check if command exists
check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        error "Command not found: $cmd"
        return 1
    fi
    return 0
}

# Validate Python environment
validate_environment() {
    log "Validating Python environment..."
    
    # Check Python
    if ! check_command python3; then
        error "Python3 is required but not installed"
        return 1
    fi
    
    # Check required Python packages
    local packages=("torch" "optuna" "yaml" "numpy" "tensorboard")
    for pkg in "${packages[@]}"; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            error "Required Python package not found: $pkg"
            return 1
        fi
    done
    
    info "Python environment validated successfully"
    return 0
}

# Validate configuration files
validate_configs() {
    log "Validating configuration files..."
    
    if ! check_file "$DEFAULT_CONFIG"; then
        error "Default config not found: $DEFAULT_CONFIG"
        return 1
    fi
    
    if ! check_file "$SEARCH_SPACE_CONFIG"; then
        error "Search space config not found: $SEARCH_SPACE_CONFIG"
        return 1
    fi
    
    if ! check_file "$HPO_SCRIPT"; then
        error "HPO script not found: $HPO_SCRIPT"
        return 1
    fi
    
    # Validate YAML syntax
    if ! python3 -c "import yaml; yaml.safe_load(open('$DEFAULT_CONFIG'))" 2>/dev/null; then
        error "Invalid YAML syntax in: $DEFAULT_CONFIG"
        return 1
    fi
    
    if ! python3 -c "import yaml; yaml.safe_load(open('$SEARCH_SPACE_CONFIG'))" 2>/dev/null; then
        error "Invalid YAML syntax in: $SEARCH_SPACE_CONFIG"
        return 1
    fi
    
    info "Configuration files validated successfully"
    return 0
}

# Check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L | wc -l)
        if [[ $gpu_count -gt 0 ]]; then
            info "Found $gpu_count GPU(s) available"
            nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | \
                while IFS=', ' read -r name memory_total memory_used; do
                    local memory_free=$((memory_total - memory_used))
                    info "GPU: $name, Memory: ${memory_free}MB free / ${memory_total}MB total"
                done
        else
            warn "No GPUs detected, will use CPU"
        fi
    else
        warn "nvidia-smi not found, GPU status unknown"
    fi
}

# Monitor system resources
monitor_resources() {
    local pid="$1"
    local log_file="$2"
    
    if [[ "$MONITOR_GPU" == true ]] && command -v nvidia-smi &> /dev/null; then
        (
            while kill -0 "$pid" 2>/dev/null; do
                echo "$(date +'%Y-%m-%d %H:%M:%S') GPU:" >> "$log_file"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits >> "$log_file"
                sleep 60
            done
        ) &
    fi
    
    if [[ "$MONITOR_MEMORY" == true ]]; then
        (
            while kill -0 "$pid" 2>/dev/null; do
                echo "$(date +'%Y-%m-%d %H:%M:%S') Memory:" >> "$log_file"
                if command -v free >/dev/null 2>&1; then
                    free -h | head -2 >> "$log_file"
                elif [[ "$(uname)" == "Darwin" ]]; then
                    vm_stat | sed -n '1,6p' >> "$log_file"
                else
                    echo "Memory monitoring unavailable on this platform" >> "$log_file"
                fi
                sleep 60
            done
        ) &
    fi
}

# Create timestamped results directory
create_results_dir() {
    local base_dir="$1"
    local timestamp=$(date +'%Y%m%d_%H%M%S')
    local results_dir="${base_dir}/${timestamp}"
    
    mkdir -p "$results_dir"
    echo "$results_dir"
}

# =============================================================================
# HPO Mode Functions
# =============================================================================

# Basic HPO Mode - Quick optimization focusing on core parameters
run_basic_hpo() {
    local trials=${1:-30}
    local timeout=${2:-1200}  # 20 minutes per trial
    local save_dir=${3:-"hpo_results_basic"}
    local search_space_path=${4:-"configs/hpo_search_space_basic.yaml"}
    
    log "Starting Basic HPO Mode"
    info "Trials: $trials, Timeout: ${timeout}s per trial"
    info "Focus: Core parameters (learning_rate, d_model, batch_size, dropout)"
    
    local results_dir=$(create_results_dir "$save_dir")
    local log_file="$results_dir/hpo_basic.log"
    
    log "Results will be saved to: $results_dir"
    log "Log file: $log_file"
    
    # Create temporary config with 3-fold CV for basic HPO
    local basic_config="$results_dir/basic_config.yaml"
    cp "$DEFAULT_CONFIG" "$basic_config"
    python3 - <<PY
import yaml
cfg_path = r"$basic_config"
with open(cfg_path) as f: cfg = yaml.safe_load(f)
cfg.setdefault('cross_validation', {})['n_folds'] = 3
with open(cfg_path, 'w') as f: yaml.safe_dump(cfg, f)
PY
    
    # Run basic HPO
    python3 "$HPO_SCRIPT" \
        --config "$basic_config" \
        --mode hpo \
        --n_trials "$trials" \
        --timeout "$timeout" \
        --sampler tpe \
        --pruner median \
        --save_dir "$results_dir" \
        --cv_folds 3 \
        --search_space_path "$search_space_path" \
        --log_level "$DEFAULT_LOG_LEVEL" \
        > >(tee "$log_file") 2>&1 &
    
    local hpo_pid=$!
    monitor_resources "$hpo_pid" "$results_dir/resources.log"
    
    # Wait for completion
    if wait "$hpo_pid"; then
        log "Basic HPO completed successfully"
        info "Results saved to: $results_dir"
        
        # Quick analysis
        if [[ -f "$results_dir/hpo_results.json" ]]; then
            python3 "$HPO_SCRIPT" --mode analyze --results_dir "$results_dir" \
                2>&1 | tee "$results_dir/analysis.log"
        fi
        
        return 0
    else
        error "Basic HPO failed"
        return 1
    fi
}

# Comprehensive HPO Mode - Full optimization using external search space
run_full_hpo() {
    local trials=${1:-100}
    local timeout=${2:-1800}  # 30 minutes per trial
    local save_dir=${3:-"hpo_results_full"}
    
    log "Starting Comprehensive HPO Mode"
    info "Trials: $trials, Timeout: ${timeout}s per trial"
    info "Using comprehensive search space: $SEARCH_SPACE_CONFIG"
    info "Cross-validation: 5 folds with patient-level stratification"
    
    local results_dir=$(create_results_dir "$save_dir")
    local log_file="$results_dir/hpo_full.log"
    
    log "Results will be saved to: $results_dir"
    log "Log file: $log_file"
    
    # Check available disk space (need ~50GB for full HPO)
    local required_space_gb=50
    local available_space_gb=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    if [[ $available_space_gb -lt $required_space_gb ]]; then
        warn "Low disk space: ${available_space_gb}GB available, ${required_space_gb}GB recommended"
    fi
    
    # Run comprehensive HPO
    python3 "$HPO_SCRIPT" \
        --config "$DEFAULT_CONFIG" \
        --search_space_path "$SEARCH_SPACE_CONFIG" \
        --mode hpo \
        --n_trials "$trials" \
        --timeout "$timeout" \
        --sampler tpe \
        --pruner median \
        --save_dir "$results_dir" \
        --cv_folds 5 \
        --log_level "$DEFAULT_LOG_LEVEL" \
        > >(tee "$log_file") 2>&1 &
    
    local hpo_pid=$!
    monitor_resources "$hpo_pid" "$results_dir/resources.log"
    
    # Wait for completion with progress updates
    local start_time=$(date +%s)
    while kill -0 "$hpo_pid" 2>/dev/null; do
        sleep 300  # Check every 5 minutes
        local elapsed=$(($(date +%s) - start_time))
        local hours=$((elapsed / 3600))
        local minutes=$(((elapsed % 3600) / 60))
        info "HPO running for ${hours}h ${minutes}m..."
    done
    
    if wait "$hpo_pid"; then
        log "Comprehensive HPO completed successfully"
        local end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        local total_hours=$((total_time / 3600))
        local total_minutes=$(((total_time % 3600) / 60))
        info "Total optimization time: ${total_hours}h ${total_minutes}m"
        info "Results saved to: $results_dir"
        
        # Comprehensive analysis
        if [[ -f "$results_dir/hpo_results.json" ]]; then
            log "Generating comprehensive analysis..."
            python3 "$HPO_SCRIPT" --mode analyze --results_dir "$results_dir" \
                2>&1 | tee "$results_dir/analysis.log"
        fi
        
        return 0
    else
        error "Comprehensive HPO failed"
        return 1
    fi
}

# AutoML Mode - Automated machine learning pipeline
run_automl() {
    local trials=${1:-50}
    local save_dir=${2:-"automl_results"}
    
    log "Starting AutoML Mode"
    info "Trials: $trials"
    info "Features: Automated feature engineering, ensemble methods"
    
    local results_dir=$(create_results_dir "$save_dir")
    local log_file="$results_dir/automl.log"
    
    log "Results will be saved to: $results_dir"
    
    # Run AutoML
    python3 "$HPO_SCRIPT" \
        --config "$DEFAULT_CONFIG" \
        --mode automl \
        --n_trials "$trials" \
        --save_dir "$results_dir" \
        --cv_folds 5 \
        --log_level "$DEFAULT_LOG_LEVEL" \
        > >(tee "$log_file") 2>&1 &
    
    local automl_pid=$!
    monitor_resources "$automl_pid" "$results_dir/resources.log"
    
    if wait "$automl_pid"; then
        log "AutoML completed successfully"
        info "Results saved to: $results_dir"
        return 0
    else
        error "AutoML failed"
        return 1
    fi
}

# Multi-objective Optimization - Balance accuracy vs efficiency
run_multi_objective() {
    local trials=${1:-75}
    local timeout=${2:-1800}
    local save_dir=${3:-"hpo_results_multi_obj"}
    
    log "Starting Multi-objective Optimization"
    info "Trials: $trials, Timeout: ${timeout}s per trial"
    info "Objectives: Maximize validation score, Minimize training time"
    info "Analysis: Pareto frontier optimization"
    
    local results_dir=$(create_results_dir "$save_dir")
    local log_file="$results_dir/hpo_multi_obj.log"
    
    log "Results will be saved to: $results_dir"
    
    # Create multi-objective configuration
    local multi_obj_config="$results_dir/multi_objective_config.yaml"
    cp "$DEFAULT_CONFIG" "$multi_obj_config"
    
    # Enable multi-objective optimization in config
    python3 - <<'PY'
import yaml
p = r"$multi_obj_config"
with open(p) as f: cfg = yaml.safe_load(f)
hpo = cfg.setdefault('hyperparameter_optimization', {})
mo = hpo.setdefault('multi_objective', {})
mo['enabled'] = True
mo['objectives'] = ['val_combined_score', 'total_training_time']
mo['directions'] = ['maximize', 'minimize']
with open(p, 'w') as f: yaml.safe_dump(cfg, f)
PY
    
    # Run multi-objective HPO
    python3 "$HPO_SCRIPT" \
        --config "$multi_obj_config" \
        --search_space_path "$SEARCH_SPACE_CONFIG" \
        --mode hpo \
        --n_trials "$trials" \
        --timeout "$timeout" \
        --sampler tpe \
        --pruner median \
        --save_dir "$results_dir" \
        --cv_folds 5 \
        --log_level "$DEFAULT_LOG_LEVEL" \
        > >(tee "$log_file") 2>&1 &
    
    local hpo_pid=$!
    monitor_resources "$hpo_pid" "$results_dir/resources.log"
    
    if wait "$hpo_pid"; then
        log "Multi-objective HPO completed successfully"
        info "Results saved to: $results_dir"
        
        # Generate Pareto frontier analysis
        if [[ -f "$results_dir/hpo_results.json" ]]; then
            log "Generating Pareto frontier analysis..."
            python3 "$HPO_SCRIPT" --mode analyze --results_dir "$results_dir" \
                2>&1 | tee "$results_dir/pareto_analysis.log"
        fi
        
        return 0
    else
        error "Multi-objective HPO failed"
        return 1
    fi
}

# Analysis Mode - Analyze existing HPO results
run_analysis() {
    local results_dir=${1:-""}
    
    if [[ -z "$results_dir" ]]; then
        latest_dir=$(ls -td hpo_results*/**/ automl_results*/**/ 2>/dev/null | head -1)
        if [[ -n "$latest_dir" ]]; then
            results_dir="${latest_dir%/}"
            log "Auto-detected latest results directory: $results_dir"
        else
            error "No results directory found and none specified"
            error "Usage: $0 analysis <results_directory>"
            return 1
        fi
    fi
    
    if [[ ! -d "$results_dir" ]]; then
        error "Results directory not found: $results_dir"
        return 1
    fi
    
    if [[ ! -f "$results_dir/hpo_results.json" ]]; then
        error "No HPO results file found in: $results_dir"
        return 1
    fi
    
    log "Starting Analysis Mode"
    info "Analyzing results from: $results_dir"
    
    # Run comprehensive analysis
    python3 "$HPO_SCRIPT" --mode analyze --results_dir "$results_dir" \
        2>&1 | tee "$results_dir/detailed_analysis.log"
    
    # Generate additional visualizations if matplotlib is available
    if python3 -c "import matplotlib" 2>/dev/null; then
        log "Generating additional visualizations..."
        
        # Create analysis script
        cat > "$results_dir/generate_plots.py" << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = Path("hpo_results.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    # Plot optimization history
    history = results.get('optimization_history', [])
    if history:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history)
        plt.title('Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('Objective Value')
        plt.grid(True, alpha=0.3)
        
        # Parameter importance
        importance = results.get('parameter_importance', {})
        if importance:
            plt.subplot(2, 2, 2)
            params = list(importance.keys())[:10]  # Top 10
            values = [importance[p] for p in params]
            plt.barh(range(len(params)), values)
            plt.yticks(range(len(params)), params)
            plt.title('Top 10 Parameter Importance')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
        
        # Best parameters
        best_params = results.get('best_params', {})
        if best_params:
            plt.subplot(2, 2, 3)
            # Create a summary text plot
            plt.text(0.1, 0.9, f"Best Objective: {results.get('best_value', 'N/A'):.6f}", 
                    transform=plt.gca().transAxes, fontsize=12, weight='bold')
            
            text = "Best Parameters:\n"
            for i, (param, value) in enumerate(best_params.items()):
                if i < 10:  # Limit to top 10
                    text += f"{param}: {value}\n"
            
            plt.text(0.1, 0.8, text, transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top')
            plt.axis('off')
            plt.title('Best Configuration')
        
        plt.tight_layout()
        plt.savefig('hpo_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Analysis plots saved to: hpo_analysis_plots.png")
EOF
        
        # Run analysis script
        cd "$results_dir"
        python3 generate_plots.py
        cd - > /dev/null
        
        if [[ -f "$results_dir/hpo_analysis_plots.png" ]]; then
            info "Analysis plots saved to: $results_dir/hpo_analysis_plots.png"
        fi
    fi
    
    log "Analysis completed successfully"
    return 0
}

# Resume Mode - Resume interrupted optimization
run_resume() {
    local study_dir=${1:-""}
    local additional_trials=${2:-50}
    
    if [[ -z "$study_dir" ]]; then
        error "Please specify the study directory to resume"
        echo "Usage: $0 resume [study_directory] [additional_trials]"
        return 1
    fi
    
    if [[ ! -d "$study_dir" ]]; then
        error "Study directory not found: $study_dir"
        return 1
    fi
    
    log "Starting Resume Mode"
    info "Resuming from: $study_dir"
    info "Additional trials: $additional_trials"
    
    # Create resume configuration
    local resume_config="$study_dir/resume_config.yaml"
    if [[ -f "$study_dir/config.yaml" ]]; then
        cp "$study_dir/config.yaml" "$resume_config"
    else
        cp "$DEFAULT_CONFIG" "$resume_config"
    fi
    
    # Resume optimization
    local resume_log="$study_dir/resume_$(date +'%Y%m%d_%H%M%S').log"
    python3 "$HPO_SCRIPT" \
        --config "$resume_config" \
        --search_space_path "$SEARCH_SPACE_CONFIG" \
        --mode hpo \
        --n_trials "$additional_trials" \
        --timeout 1800 \
        --sampler tpe \
        --pruner median \
        --save_dir "$study_dir" \
        --cv_folds 5 \
        --log_level "$DEFAULT_LOG_LEVEL" \
        > >(tee "$resume_log") 2>&1 &
    
    local resume_pid=$!
    monitor_resources "$resume_pid" "$study_dir/resume_resources.log"
    
    if wait "$resume_pid"; then
        log "Resume completed successfully"
        info "Updated results saved to: $study_dir"
        
        # Run fresh analysis
        run_analysis "$study_dir"
        return 0
    else
        error "Resume failed"
        return 1
    fi
}

# =============================================================================
# Main Script Logic
# =============================================================================

# Display help
show_help() {
    cat << EOF
ABR Transformer Hyperparameter Optimization Script

Usage: $0 [MODE] [OPTIONS]

MODES:
    basic [trials] [timeout] [save_dir]
        Quick optimization focusing on core parameters
        Default: 30 trials, 1200s timeout, hpo_results_basic/
    
    full [trials] [timeout] [save_dir]  
        Comprehensive optimization using external search space
        Default: 100 trials, 1800s timeout, hpo_results_full/
    
    automl [trials] [save_dir]
        Automated machine learning pipeline with ensemble methods
        Default: 50 trials, automl_results/
    
    multi_obj [trials] [timeout] [save_dir]
        Multi-objective optimization (accuracy vs efficiency)
        Default: 75 trials, 1800s timeout, hpo_results_multi_obj/
    
    analysis [results_dir]
        Analyze existing HPO results and generate visualizations
        Default: auto-detect most recent results
    
    resume [study_dir] [additional_trials]
        Resume interrupted optimization from existing study
        Default: 50 additional trials

OPTIONS:
    --dry-run           Validate configuration without running optimization
    --no-gpu-monitor    Disable GPU monitoring
    --no-memory-monitor Disable memory monitoring
    --quiet            Reduce output verbosity
    --help             Show this help message

EXAMPLES:
    # Quick optimization with default parameters
    $0 basic
    
    # Comprehensive optimization with custom settings
    $0 full 150 2400 my_hpo_results
    
    # Multi-objective optimization
    $0 multi_obj 100
    
    # Analyze existing results
    $0 analysis hpo_results_full/20240101_120000
    
    # Resume interrupted optimization
    $0 resume hpo_results_full/20240101_120000 25

REQUIREMENTS:
    - Python 3.8+ with PyTorch, Optuna, and dependencies
    - CUDA-capable GPU (recommended)
    - At least 16GB RAM for full optimization
    - 50GB+ free disk space for comprehensive optimization

For detailed documentation, see: documentation/HPO_USAGE_GUIDE.md
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --no-gpu-monitor)
                MONITOR_GPU=false
                shift
                ;;
            --no-memory-monitor)
                MONITOR_MEMORY=false
                shift
                ;;
            --quiet)
                DEFAULT_LOG_LEVEL="WARNING"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            -*)
                warn "Unknown option: $1"
                shift
                ;;
            *)
                break
                ;;
        esac
    done
}

# Main function
main() {
    local mode=${1:-"help"}
    shift || true
    
    # Parse global options first
    parse_args "$@"
    
    # Print header
    echo
    log "ABR Transformer Hyperparameter Optimization"
    echo "=============================================="
    echo
    
    # Validate environment unless showing help
    if [[ "$mode" != "help" ]] && [[ "$mode" != "analysis" ]]; then
        if ! validate_environment || ! validate_configs; then
            error "Environment validation failed"
            exit 1
        fi
        
        check_gpu
        echo
    fi
    
    # Handle dry run
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log "Dry run mode - configuration validated successfully"
        exit 0
    fi
    
    # Route to appropriate mode
    case "$mode" in
        basic)
            run_basic_hpo "$@"
            ;;
        full)
            run_full_hpo "$@"
            ;;
        automl)
            run_automl "$@"
            ;;
        multi_obj|multi-obj|multi_objective)
            run_multi_objective "$@"
            ;;
        analysis|analyze)
            run_analysis "$@"
            ;;
        resume)
            run_resume "$@"
            ;;
        help|--help|-h|"")
            show_help
            exit 0
            ;;
        *)
            error "Unknown mode: $mode"
            show_help
            exit 1
            ;;
    esac
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo
        log "HPO operation completed successfully!"
        info "Check the results directory for outputs and analysis"
        echo
    else
        echo
        error "HPO operation failed with exit code: $exit_code"
        error "Check the log files for detailed error information"
        echo
        exit $exit_code
    fi
}

# Run main function with all arguments
main "$@"
