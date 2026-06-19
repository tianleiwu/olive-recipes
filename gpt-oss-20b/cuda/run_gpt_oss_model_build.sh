#!/bin/bash
set -e

# ============================================================================
# gpt-oss-20b CUDA Model Build & Benchmark Script
# ============================================================================
# This script automates:
#   1. Python venv preparation
#   2. Building 4 QMoE model variants using Olive
#   3. Benchmarking each variant
#   4. Generating a summary comparison table
# ============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${HOME}/venv"
CUDA_HOME="${HOME}/cuda13.0"
CUDNN_HOME="${HOME}/cudnn_9.19_cuda13"
ORT_HOME="${HOME}/ort_home_cu130"
GENAI_DIR="${HOME}/onnxruntime-genai"
OLIVE_RECIPES="${HOME}/olive-recipes"
VARIANTS_DIR="${SCRIPT_DIR}/variants"
BENCHMARK_LOG="${SCRIPT_DIR}/benchmark_results.txt"

PROMPT="What is the capital of France?"
MAX_LEN=256

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Step 1: Venv Preparation
# ============================================================================

step_venv_preparation() {
    log_info "===== Step 1: Python venv Preparation ====="
    
    if [ -d "$VENV_PATH" ]; then
        log_info "venv already exists at $VENV_PATH"
    else
        log_info "Creating venv at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
        log_info "venv created successfully"
    fi
    
    log_info "Activating venv..."
    source "$VENV_PATH/bin/activate"
    
    log_info "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel -q
    
    log_info "Installing required packages..."
    pip install torch onnx onnxruntime-genai neural-compressor olive-ai -q
    
    log_info "venv preparation complete"
    echo ""
}

# ============================================================================
# Step 2: Setup Environment
# ============================================================================

setup_environment() {
    log_info "===== Setting up CUDA/cuDNN environment ====="
    
    export CUDA_HOME
    export CUDNN_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$ORT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
    
    log_info "CUDA_HOME: $CUDA_HOME"
    log_info "CUDNN_HOME: $CUDNN_HOME"
    log_info "ORT_HOME: $ORT_HOME"
    log_info "GENAI_DIR: $GENAI_DIR"
    
    # Verify CUDA is available
    if ! command -v nvcc &> /dev/null; then
        log_error "CUDA compiler (nvcc) not found. Check CUDA_HOME path."
        exit 1
    fi
    
    log_info "CUDA setup verified: $(nvcc --version | head -n 1)"
    echo ""
}

# ============================================================================
# Step 3: Build Model Variants
# ============================================================================

build_variants() {
    log_info "===== Step 2: Building Model Variants ====="
    
    cd "$SCRIPT_DIR"
    
    local configs=(
        "gpt-oss-20b_cuda_int4_int4_qmoe_default.json"
        "gpt-oss-20b_cuda_int4_int4_qmoe_k_quant_mixed.json"
        "gpt-oss-20b_cuda_int4_int8_qmoe_default.json"
        "gpt-oss-20b_cuda_int4_int8_qmoe_k_quant_mixed.json"
    )
    
    local variant_names=(
        "cuda_int4_int4_qmoe_default"
        "cuda_int4_int4_qmoe_k_quant_mixed"
        "cuda_int4_int8_qmoe_default"
        "cuda_int4_int8_qmoe_k_quant_mixed"
    )
    
    mkdir -p "$VARIANTS_DIR"
    
    for i in "${!configs[@]}"; do
        local config="${configs[$i]}"
        local variant="${variant_names[$i]}"
        
        if [ ! -f "$config" ]; then
            log_error "Config file not found: $config"
            exit 1
        fi
        
        log_info "Building variant: $variant (Config: $config)"
        
        start_time=$(date +%s)
        olive run --config "$config"
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        log_info "Build completed for $variant (Duration: ${duration}s)"
        
        # Save to variants directory
        rm -rf "$VARIANTS_DIR/$variant"
        cp -a model "$VARIANTS_DIR/$variant"
        
        # CRITICAL FIX: Enable CUDA graph for optimal throughput
        # CUDA graph captures kernel submissions to reduce launch overhead
        # Without this, generation throughput is 5-6x slower (50 TPS vs 275 TPS)
        local config_file="$VARIANTS_DIR/$variant/genai_config.json"
        sed -i 's/"enable_cuda_graph": "0"/"enable_cuda_graph": "1"/' "$config_file"
        
        log_info "Saved to: $VARIANTS_DIR/$variant"
        log_info "✓ CUDA graph enabled (enable_cuda_graph=1 for optimal decode performance)"
        echo ""
    done
    
    log_info "All model variants built successfully"
    log_info "Available variants:"
    ls -1 "$VARIANTS_DIR"
    echo ""
}

# ============================================================================
# Step 4: Run Benchmarks
# ============================================================================

run_benchmarks() {
    log_info "===== Step 3: Running Benchmarks ====="
    
    echo "===== gpt-oss-20b CUDA Benchmark $(date) =====" | tee "$BENCHMARK_LOG"
    echo "Prompt: '$PROMPT'  max_length=$MAX_LEN  EP=cuda" | tee -a "$BENCHMARK_LOG"
    echo "" | tee -a "$BENCHMARK_LOG"
    
    cd "$GENAI_DIR"
    
    local variant_names=(
        "cuda_int4_int4_qmoe_default"
        "cuda_int4_int4_qmoe_k_quant_mixed"
        "cuda_int4_int8_qmoe_default"
        "cuda_int4_int8_qmoe_k_quant_mixed"
    )
    
    for variant in "${variant_names[@]}"; do
        local model_path="$VARIANTS_DIR/$variant"
        
        if [ ! -d "$model_path" ]; then
            log_error "Model path not found: $model_path"
            exit 1
        fi
        
        log_info "Benchmarking: $variant"
        echo "---------- $variant ----------" | tee -a "$BENCHMARK_LOG"
        
        python examples/python/model-qa.py \
            -m "$model_path" \
            -e cuda \
            --non_interactive \
            -up "$PROMPT" \
            -g \
            -l "$MAX_LEN" 2>&1 | tee -a "$BENCHMARK_LOG"
        
        echo "" | tee -a "$BENCHMARK_LOG"
    done
    
    log_info "All benchmarks completed"
    echo ""
}

# ============================================================================
# Step 5: Generate Summary
# ============================================================================

generate_summary() {
    log_info "===== Step 4: Generating Summary ====="
    
    echo "===== SUMMARY =====" | tee -a "$BENCHMARK_LOG"
    echo "" | tee -a "$BENCHMARK_LOG"
    
    log_info "Extracting metrics from benchmark log..."
    
    # Extract metrics using grep and format as table
    {
        echo "| Variant | MoE Precision | Quant Algorithm | TTFT (s) | Prompt TPS | Gen TPS |"
        echo "|---------|---------------|-----------------|----------|-----------|---------|"
        
        grep -A 1 "^---------- cuda_int4_int4_qmoe_default" "$BENCHMARK_LOG" | grep "Time to first" | awk '{print "| cuda_int4_int4_default | INT4 | RTN |", $NF}' || true
        grep -A 1 "^---------- cuda_int4_int4_qmoe_k_quant_mixed" "$BENCHMARK_LOG" | grep "Time to first" | awk '{print "| cuda_int4_int4_k_quant | INT4 | k-quant |", $NF}' || true
        grep -A 1 "^---------- cuda_int4_int8_qmoe_default" "$BENCHMARK_LOG" | grep "Time to first" | awk '{print "| cuda_int4_int8_default | INT8 | RTN |", $NF}' || true
        grep -A 1 "^---------- cuda_int4_int8_qmoe_k_quant_mixed" "$BENCHMARK_LOG" | grep "Time to first" | awk '{print "| cuda_int4_int8_k_quant | INT8 | k-quant |", $NF}' || true
    } | tee -a "$BENCHMARK_LOG"
    
    echo "" | tee -a "$BENCHMARK_LOG"
    log_info "Benchmark log saved to: $BENCHMARK_LOG"
    
    log_info "Script execution complete! ✓"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log_info "Starting gpt-oss-20b CUDA Model Build & Benchmark"
    log_info "Script directory: $SCRIPT_DIR"
    echo ""
    
    step_venv_preparation
    setup_environment
    build_variants
    run_benchmarks
    generate_summary+
    
    log_info "All steps completed successfully!"
    echo ""
}

# Execute main function
main "$@"
