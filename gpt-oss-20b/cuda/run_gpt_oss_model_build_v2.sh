#!/bin/bash
set -e

# ============================================================================
# gpt-oss-20b CUDA Model Build, Benchmark & MMLU Eval Script (v2)
# ============================================================================
# This script automates the full experiment for 4 QMoE model variants:
#   1. Python venv preparation
#   2. Installing onnxruntime-genai BUILT FROM SOURCE (so the locally patched
#      model-builder is used by Olive's ModelBuilder pass)
#   3. Building 4 QMoE model variants using Olive
#   4. Benchmarking decode throughput for each variant (XQA enabled)
#   5. Evaluating full MMLU accuracy for each variant (multi-GPU shard runner)
#   6. Generating a summary table and appending a record to
#      gpt-oss-20b/gpt_oss_20b_experiments.md
#
# Most paths/knobs below can be overridden via environment variables.
# ============================================================================

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Canonical venv: .venv_cu130 holds the from-source ONNX Runtime + genai build.
# (Note: ~/venv's console scripts have shebangs pointing here, so this is the
#  single source of truth. We invoke its binaries by absolute path.)
VENV_PATH="${VENV_PATH:-${HOME}/onnxruntime/.venv_cu130}"
VPY="$VENV_PATH/bin/python"
VPIP="$VENV_PATH/bin/pip"
VOLIVE="$VENV_PATH/bin/olive"
CUDA_HOME="${CUDA_HOME:-${HOME}/cuda13.0}"
CUDNN_HOME="${CUDNN_HOME:-${HOME}/cudnn9.19_cuda13}"
ORT_DIR="${ORT_DIR:-${HOME}/onnxruntime}"
ORT_HOME="${ORT_HOME:-${HOME}/ort_home_cu130}"
GENAI_DIR="${GENAI_DIR:-${HOME}/onnxruntime-genai}"
OLIVE_RECIPES="${OLIVE_RECIPES:-${HOME}/olive-recipes}"
EVALS_DIR="${EVALS_DIR:-${HOME}/evals}"
VARIANTS_DIR="${VARIANTS_DIR:-${SCRIPT_DIR}/variants}"
BENCHMARK_LOG="${BENCHMARK_LOG:-${SCRIPT_DIR}/benchmark_results_v2.txt}"
RESULTS_TSV="${RESULTS_TSV:-${SCRIPT_DIR}/experiment_results_v2.tsv}"
EXPERIMENTS_MD="${EXPERIMENTS_MD:-${OLIVE_RECIPES}/gpt-oss-20b/gpt_oss_20b_experiments.md}"

# Multi-GPU MMLU runner (proven shard pipeline). Override via MMLU_RUNNER.
MMLU_RUNNER="${MMLU_RUNNER:-${HOME}/dev/scripts/h200_18/run_evals_parallel.sh}"

# ---- Build-from-source knobs ----
ORT_BUILD_DIR="${ORT_BUILD_DIR:-build/cu130}"     # relative to ORT_DIR
GENAI_BUILD_DIR="${GENAI_BUILD_DIR:-build/cu130}" # relative to GENAI_DIR
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDAARCHS="${CUDAARCHS:-90}"                       # H200 = sm_90
FORCE_BUILD_GENAI="${FORCE_BUILD_GENAI:-0}"        # 1 = rebuild genai wheel from source

# ---- Benchmark knobs (decode throughput, XQA enabled) ----
PROMPT="${PROMPT:-What is the capital of France?}"
MAX_LEN="${MAX_LEN:-256}"
BATCH="${BATCH:-1}"
PROMPT_LEN="${PROMPT_LEN:-512}"
GEN_LEN="${GEN_LEN:-128}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-2}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"
XQA="${XQA:-1}"                                    # ORT_ENABLE_XQA for decode
SYNC_PROVIDER_LIB="${SYNC_PROVIDER_LIB:-1}"        # sync freshly built CUDA provider into venv

# ---- MMLU knobs ----
RUN_MMLU="${RUN_MMLU:-1}"
MMLU_MAX_SAMPLES="${MMLU_MAX_SAMPLES:-0}"          # 0 = full 14042-sample set
MMLU_PARALLEL_PER_GPU="${MMLU_PARALLEL_PER_GPU:-2}"
MMLU_GPUS="${MMLU_GPUS:-}"                         # empty = all detected GPUs
EVAL_RUNS_DIR="${EVAL_RUNS_DIR:-${HOME}/eval_runs}"

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
    "$VPIP" install --upgrade pip setuptools wheel -q
    
    # NOTE: We intentionally do NOT install the prebuilt onnxruntime-genai here.
    # The from-source ONNX Runtime + genai (with the patched model builder) are
    # installed by install_ort_from_source() and ensure_genai_from_source().
    log_info "Installing required packages (Olive + eval deps)..."
    "$VPIP" install onnx neural-compressor olive-ai -q
    "$VPIP" install requests accelerate -q
    log_info "venv preparation complete"
    echo ""
}

# ============================================================================
# Step 1a: Install ONNX Runtime built from source (latest, e.g. 1.28.0)
# ============================================================================
# The venv may ship an older pip onnxruntime (1.27.0). XQA decode + the matching
# CUDA provider live in the from-source build (1.28.0). Install the newest
# onnxruntime_gpu wheel produced under the ORT build dist so the runtime matches
# the provider lib we later sync into the package.
# ============================================================================

install_ort_from_source() {
    log_info "===== Step 1a: Install ONNX Runtime (from source) ====="
    local dist="$ORT_DIR/$ORT_BUILD_DIR/$BUILD_TYPE/dist"
    local wheel
    # Pick the highest-version onnxruntime_gpu wheel available.
    wheel="$(ls -1 "$dist"/onnxruntime_gpu-*.whl 2>/dev/null | sort -V | tail -1 || true)"
    if [ -z "$wheel" ]; then
        log_warn "No from-source onnxruntime_gpu wheel found in $dist; keeping installed onnxruntime"
        return 0
    fi
    local cur
    cur="$("$VPY" -c 'import onnxruntime as ort; print(ort.__version__)' 2>/dev/null || echo none)"
    log_info "Installed onnxruntime: $cur ; installing from-source wheel: $(basename "$wheel")"
    "$VPIP" install "$wheel" --force-reinstall --no-deps -q
    log_info "onnxruntime now: $("$VPY" -c 'import onnxruntime as ort; print(ort.__version__)')"
    echo ""
}

# ============================================================================
# Step 2: Setup Environment
# ============================================================================

setup_environment() {
    log_info "===== Setting up CUDA/cuDNN environment ====="
    
    export CUDA_HOME
    export CUDNN_HOME
    export CUDA_PATH="$CUDA_HOME"
    export CUDAARCHS
    export ORT_ENABLE_XQA="$XQA"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$ORT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
    
    log_info "CUDA_HOME: $CUDA_HOME"
    log_info "CUDNN_HOME: $CUDNN_HOME"
    log_info "ORT_DIR:   $ORT_DIR"
    log_info "ORT_HOME:  $ORT_HOME"
    log_info "GENAI_DIR: $GENAI_DIR"
    log_info "ORT_ENABLE_XQA: $ORT_ENABLE_XQA"
    
    # Verify CUDA is available
    if ! command -v nvcc &> /dev/null; then
        log_error "CUDA compiler (nvcc) not found. Check CUDA_HOME path."
        exit 1
    fi
    
    log_info "CUDA setup verified: $(nvcc --version | head -n 1)"
    echo ""
}

# ============================================================================
# Step 1b: Install onnxruntime-genai built from source
# ============================================================================
# The model builder used by Olive's ModelBuilder pass ships inside the
# onnxruntime-genai Python package. We have local changes to that builder, so we
# must install the from-source genai wheel (not the prebuilt PyPI package).
# Behaviour:
#   - If a from-source wheel exists under $GENAI_DIR/$GENAI_BUILD_DIR and
#     FORCE_BUILD_GENAI=0, install it (reuse existing build).
#   - If no wheel exists or FORCE_BUILD_GENAI=1, build genai from source against
#     the assembled $ORT_HOME, then install the produced wheel.
# ============================================================================

assemble_ort_home() {
    local ort_build="$ORT_DIR/$ORT_BUILD_DIR/$BUILD_TYPE"
    [ -f "$ort_build/libonnxruntime.so" ] || {
        log_error "ONNX Runtime libs not found in $ort_build. Build ONNX Runtime first."
        exit 1
    }
    log_info "Assembling ORT_HOME at $ORT_HOME"
    rm -rf "$ORT_HOME"
    mkdir -p "$ORT_HOME/include" "$ORT_HOME/lib"
    cp -f "$ORT_DIR"/include/onnxruntime/core/session/*.h "$ORT_HOME/include/" 2>/dev/null || true
    cp -f "$ORT_DIR"/include/onnxruntime/core/session/*.inc "$ORT_HOME/include/" 2>/dev/null || true
    cp -f "$ORT_DIR"/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h "$ORT_HOME/include/" 2>/dev/null || true
    cp -Pf "$ort_build"/libonnxruntime.so* "$ORT_HOME/lib/" 2>/dev/null || true
    cp -Pf "$ort_build"/libonnxruntime_providers_shared.so "$ORT_HOME/lib/" 2>/dev/null || true
    cp -Pf "$ort_build"/libonnxruntime_providers_cuda.so "$ORT_HOME/lib/" 2>/dev/null || true
    [ -f "$ORT_HOME/lib/libonnxruntime.so" ] || {
        log_error "ORT_HOME assembly incomplete (no libonnxruntime.so)"
        exit 1
    }
}

build_genai_from_source() {
    log_info "Building onnxruntime-genai from source ($GENAI_DIR)"
    assemble_ort_home
    ( cd "$GENAI_DIR" && "$VPY" build.py \
        --use_cuda \
        --cuda_home "$CUDA_HOME" \
        --ort_home "$ORT_HOME" \
        --config "$BUILD_TYPE" \
        --build_dir "$GENAI_BUILD_DIR" \
        --parallel \
        --skip_tests \
        --skip_examples ) || {
        log_error "onnxruntime-genai build failed"
        exit 1
    }
}

ensure_genai_from_source() {
    log_info "===== Step 1b: Install onnxruntime-genai (from source) ====="

    local wheel
    wheel="$(ls -t "$GENAI_DIR/$GENAI_BUILD_DIR/$BUILD_TYPE"/wheel/onnxruntime_genai_cuda-*.whl 2>/dev/null | head -1 || true)"

    if [ "$FORCE_BUILD_GENAI" = "1" ] || [ -z "$wheel" ]; then
        if [ "$FORCE_BUILD_GENAI" = "1" ]; then
            log_info "FORCE_BUILD_GENAI=1 -> rebuilding genai wheel from source"
        else
            log_warn "No from-source genai wheel found; building from source"
        fi
        build_genai_from_source
        wheel="$(ls -t "$GENAI_DIR/$GENAI_BUILD_DIR/$BUILD_TYPE"/wheel/onnxruntime_genai_cuda-*.whl 2>/dev/null | head -1 || true)"
    else
        log_info "Reusing existing from-source genai wheel"
    fi

    [ -n "$wheel" ] || { log_error "genai wheel not found under $GENAI_DIR/$GENAI_BUILD_DIR"; exit 1; }
    log_info "Installing genai wheel: $(basename "$wheel")"
    "$VPIP" install "$wheel" --force-reinstall --no-deps -q
    log_info "onnxruntime-genai version: $("$VPY" -c 'import onnxruntime_genai as og; print(og.__version__)')"
    echo ""
}

# ============================================================================
# Step 3: Build Model Variants
# ============================================================================

# Shared variant config <-> name mapping used by build/benchmark/mmlu/summary.
CONFIGS=(
    "gpt-oss-20b_cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs0.json"
    "gpt-oss-20b_cuda_int4_int4_qmoe_rtn_last_matmul_only_qknorm_bs0.json"
    "gpt-oss-20b_cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs64.json"
    "gpt-oss-20b_cuda_int4_int4_qmoe_rtn_mixed_matmul_only_qknorm_bs0.json"
)
VARIANT_NAMES=(
    "cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs0"
    "cuda_int4_int4_qmoe_rtn_last_matmul_only_qknorm_bs0"
    "cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs64"
    "cuda_int4_int4_qmoe_rtn_mixed_matmul_only_qknorm_bs0"
)
REUSE_BUILDS="${REUSE_BUILDS:-1}"   # 1 = skip variants already present in VARIANTS_DIR

build_variants() {
    log_info "===== Step 2: Building Model Variants ====="
    
    cd "$SCRIPT_DIR"
    mkdir -p "$VARIANTS_DIR"
    
    for i in "${!CONFIGS[@]}"; do
        local config="${CONFIGS[$i]}"
        local variant="${VARIANT_NAMES[$i]}"
        local dest="$VARIANTS_DIR/$variant"
        
        if [ "$REUSE_BUILDS" = "1" ] && [ -f "$dest/genai_config.json" ]; then
            log_info "Reusing existing build: $variant ($dest)"
            # Ensure CUDA graph stays enabled for decode benchmarking.
            sed -i 's/"enable_cuda_graph": "0"/"enable_cuda_graph": "1"/' "$dest/genai_config.json"
            continue
        fi
        
        if [ ! -f "$config" ]; then
            log_error "Config file not found: $config"
            exit 1
        fi
        
        log_info "Building variant: $variant (Config: $config)"
        
        start_time=$(date +%s)
        "$VOLIVE" run --config "$config"
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        log_info "Build completed for $variant (Duration: ${duration}s)"
        
        # Save to variants directory
        rm -rf "$dest"
        cp -a model "$dest"
        
        # Enable CUDA graph for optimal batch-1 decode throughput.
        sed -i 's/"enable_cuda_graph": "0"/"enable_cuda_graph": "1"/' "$dest/genai_config.json"
        
        log_info "Saved to: $dest"
        log_info "CUDA graph enabled (enable_cuda_graph=1 for optimal decode performance)"
        echo ""
    done
    
    log_info "All model variants ready"
    log_info "Available variants:"
    ls -1 "$VARIANTS_DIR"
    echo ""
}

# ============================================================================
# Step 4: Run Benchmarks (decode throughput, XQA enabled)
# ============================================================================

# Sync the freshly built CUDA provider into the venv so the benchmark exercises
# the current branch's kernels (incl. the XQA fast path).
sync_provider_lib() {
    [ "$SYNC_PROVIDER_LIB" = "1" ] || return 0
    local src="$ORT_DIR/$ORT_BUILD_DIR/$BUILD_TYPE/libonnxruntime_providers_cuda.so"
    local capi dst
    capi="$(ls -d "$VENV_PATH"/lib/python*/site-packages/onnxruntime/capi 2>/dev/null | head -1 || true)"
    [ -n "$capi" ] || { log_warn "onnxruntime capi dir not found in venv; skipping provider sync"; return 0; }
    dst="$capi/libonnxruntime_providers_cuda.so"
    if [ -f "$src" ]; then
        cp -p "$dst" "$dst.benchbak" 2>/dev/null || true
        cp "$src" "$dst"
        log_info "Synced CUDA provider lib into venv: $dst"
    else
        log_warn "Built provider lib not found ($src); using installed onnxruntime"
    fi
}

run_benchmarks() {
    log_info "===== Step 3: Running Benchmarks (decode TPS, XQA=$XQA) ====="
    
    sync_provider_lib
    
    echo "===== gpt-oss-20b CUDA decode benchmark $(date) =====" | tee "$BENCHMARK_LOG"
    echo "batch=$BATCH prompt_len=$PROMPT_LEN gen_len=$GEN_LEN reps=$REPS warmup=$WARMUP cuda_graph=$CUDA_GRAPH XQA=$XQA EP=cuda" | tee -a "$BENCHMARK_LOG"
    echo "" | tee -a "$BENCHMARK_LOG"
    
    # Reset the machine-readable results file (benchmark columns filled here).
    : > "$RESULTS_TSV"
    
    export ORT_ENABLE_XQA="$XQA"
    export CUDA_VISIBLE_DEVICES="${BENCH_GPU:-0}"
    
    for variant in "${VARIANT_NAMES[@]}"; do
        local model_path="$VARIANTS_DIR/$variant"
        
        if [ ! -d "$model_path" ]; then
            log_error "Model path not found: $model_path"
            exit 1
        fi
        
        log_info "Benchmarking: $variant"
        echo "---------- $variant ----------" | tee -a "$BENCHMARK_LOG"
        
        local csv="$SCRIPT_DIR/bench_${variant}.csv"
        local log_file="$SCRIPT_DIR/bench_${variant}.log"
        
        ( cd "$GENAI_DIR/benchmark/python" && \
          ORT_ENABLE_XQA="$XQA" "$VPY" benchmark_e2e.py \
            -i "$model_path" \
            -e cuda \
            -b "$BATCH" \
            -l "$PROMPT_LEN" \
            -g "$GEN_LEN" \
            -r "$REPS" \
            -w "$WARMUP" \
            --use_random_tokens \
            --chat_template "{input}" \
            -mn gpt-oss-20b \
            -pr int4 \
            -o "$csv" ) 2>&1 | tee "$log_file" | tee -a "$BENCHMARK_LOG"
        
        # Parse prefill / decode throughput from the benchmark log.
        local prefill decode
        prefill="$(grep -i 'Prompt Processing Throughput' "$log_file" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1 || true)"
        decode="$(grep -i 'Token Generation Throughput' "$log_file" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1 || true)"
        # variant <tab> prefill_tps <tab> decode_tps <tab> mmlu(placeholder)
        printf '%s\t%s\t%s\t\n' "$variant" "${prefill:-NA}" "${decode:-NA}" >> "$RESULTS_TSV"
        log_info "  prefill_tps=${prefill:-NA}  decode_tps=${decode:-NA}"
        echo "" | tee -a "$BENCHMARK_LOG"
    done
    
    log_info "All benchmarks completed"
    echo ""
}

# ============================================================================
# Step 5: Run full MMLU evaluation per variant (multi-GPU shard runner)
# ============================================================================

run_mmlu() {
    [ "$RUN_MMLU" = "1" ] || { log_warn "RUN_MMLU=0 -> skipping MMLU eval"; return 0; }
    log_info "===== Step 4: Running MMLU evaluation ($([ "$MMLU_MAX_SAMPLES" -gt 0 ] && echo "$MMLU_MAX_SAMPLES samples" || echo 'full set')) ====="
    
    if [ ! -x "$MMLU_RUNNER" ]; then
        log_error "MMLU runner not found/executable: $MMLU_RUNNER"
        log_error "Set MMLU_RUNNER=/path/to/run_evals_parallel.sh or RUN_MMLU=0 to skip."
        exit 1
    fi
    
    local ts; ts="$(date +%Y%m%d_%H%M%S)"
    local base_out="$EVAL_RUNS_DIR/v2_mmlu_$ts"
    mkdir -p "$base_out"
    
    local gpu_opt=()
    [ -n "$MMLU_GPUS" ] && gpu_opt=(--gpus "$MMLU_GPUS")
    local ms_opt=()
    [ "$MMLU_MAX_SAMPLES" -gt 0 ] && ms_opt=(--max-samples "$MMLU_MAX_SAMPLES")
    
    for variant in "${VARIANT_NAMES[@]}"; do
        local model_path="$VARIANTS_DIR/$variant"
        local var_out="$base_out/$variant"
        log_info "MMLU for variant: $variant"
        
        VENV="$VENV_PATH" \
        MODEL="$model_path" \
        EXECUTION_PROVIDER="cuda" \
        COMP_ARGS="model_path=$model_path,execution_provider=cuda,model_name=$variant" \
        CUDA_HOME="$CUDA_HOME" \
        CUDNN_HOME="$CUDNN_HOME" \
        ORT_ENABLE_XQA="$XQA" \
        "$MMLU_RUNNER" \
            --parallel "$MMLU_PARALLEL_PER_GPU" \
            --shard match_mmlu \
            "${gpu_opt[@]}" \
            "${ms_opt[@]}" \
            --outdir "$var_out" \
            2>&1 | tee "$base_out/${variant}.console.log" || log_warn "MMLU run for '$variant' returned non-zero"
        
        # Pool match rows from all shards -> accuracy, and store back in RESULTS_TSV.
        local acc
        acc="$(VAR_OUT="$var_out" "$VPY" - <<'PYEOF'
import glob, json, os
var_out = os.environ["VAR_OUT"]
total = correct = 0
for path in glob.glob(os.path.join(var_out, "match_mmlu_shard_*.jsonl")):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "match":
                total += 1
                if d.get("data", {}).get("correct"):
                    correct += 1
print(f"{correct/total:.4f}" if total else "NA")
PYEOF
)"
        log_info "  MMLU accuracy ($variant): $acc  (logs: $var_out)"
        # Fill the 4th column (mmlu) for this variant row in RESULTS_TSV.
        if [ -f "$RESULTS_TSV" ]; then
            local tmp; tmp="$(mktemp)"
            awk -F'\t' -v v="$variant" -v a="$acc" 'BEGIN{OFS="\t"} $1==v{$4=a} {print}' "$RESULTS_TSV" > "$tmp" && mv "$tmp" "$RESULTS_TSV"
        fi
    done
    
    MMLU_OUTDIR="$base_out"
    log_info "MMLU evaluation complete (base outdir: $base_out)"
    echo ""
}

# ============================================================================
# Step 6: Generate Summary + append record to experiments markdown
# ============================================================================

generate_summary() {
    log_info "===== Step 5: Generating Summary ====="
    
    [ -f "$RESULTS_TSV" ] || { log_warn "No results file ($RESULTS_TSV); nothing to summarize"; return 0; }
    
    RESULTS_TSV="$RESULTS_TSV" VARIANTS_DIR="$VARIANTS_DIR" \
    EXPERIMENTS_MD="$EXPERIMENTS_MD" BENCHMARK_LOG="$BENCHMARK_LOG" \
    XQA="$XQA" CUDA_GRAPH="$CUDA_GRAPH" PROMPT_LEN="$PROMPT_LEN" GEN_LEN="$GEN_LEN" \
    MMLU_MAX_SAMPLES="$MMLU_MAX_SAMPLES" \
    "$VPY" - <<'PYEOF'
import datetime, os

results = os.environ["RESULTS_TSV"]
variants_dir = os.environ["VARIANTS_DIR"]
md_path = os.environ["EXPERIMENTS_MD"]
xqa = os.environ.get("XQA", "1")
cuda_graph = os.environ.get("CUDA_GRAPH", "1")
plen = os.environ.get("PROMPT_LEN", "512")
glen = os.environ.get("GEN_LEN", "128")
ms = int(os.environ.get("MMLU_MAX_SAMPLES", "0") or "0")

# Human-readable config notes per variant (from the recipe extra_options).
NOTES = {
    "cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs0":
        "rtn int4 body (blk32), int4 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion",
    "cuda_int4_int4_qmoe_rtn_last_matmul_only_qknorm_bs0":
        "rtn int4 body (blk32), int8 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion",
    "cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs64":
        "rtn int4 body (blk64), int4 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion",
    "cuda_int4_int4_qmoe_rtn_mixed_matmul_only_qknorm_bs0":
        "rtn int4 body + mixed int8 layers, int8 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion",
}

def dir_size_gib(path):
    tot = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                tot += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return tot / (1024 ** 3) if tot else None

rows = []
with open(results, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = (line.split("\t") + ["", "", "", ""])[:4]
        variant, prefill, decode, mmlu = parts
        size = dir_size_gib(os.path.join(variants_dir, variant))
        rows.append({
            "variant": variant,
            "prefill": prefill or "NA",
            "decode": decode or "NA",
            "mmlu": mmlu or "NA",
            "size": size,
        })

def fsize(v):
    return f"{v:.2f}" if isinstance(v, float) else "n/a"

def ftps(v):
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return v or "NA"

def fmmlu(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return v or "NA"

ts = datetime.datetime.now().isoformat(timespec="seconds")
mmlu_label = f"{ms}-sample" if ms > 0 else "full (14042)"

lines = []
lines.append("")
lines.append(f"## Experiment ({ts.split('T')[0]}): rtn qk-norm-fusion variants (v2 build script)")
lines.append("")
lines.append(f"- **Generated:** {ts}")
lines.append("- **Build:** Olive recipes via `cuda/run_gpt_oss_model_build_v2.sh`, "
             "onnxruntime-genai **built from source** (patched model builder).")
lines.append(f"- **Decode TPS:** `benchmark_e2e.py`, batch 1, prompt {plen}, gen {glen}, "
             f"CUDA graph={cuda_graph}, **XQA={xqa}**.")
lines.append(f"- **MMLU:** `match_mmlu`, {mmlu_label} samples, multi-GPU shard pooled accuracy.")
lines.append("")
lines.append("| Model (variant) | Size (GiB) | Prefill TPS | Decode TPS | MMLU | Notes |")
lines.append("|---|---:|---:|---:|---:|---|")
for r in rows:
    lines.append(
        f"| `{r['variant']}` | {fsize(r['size'])} | {ftps(r['prefill'])} | "
        f"{ftps(r['decode'])} | {fmmlu(r['mmlu'])} | {NOTES.get(r['variant'], '')} |"
    )
lines.append("")

block = "\n".join(lines)

# Console echo
print("\n" + "=" * 100)
print(block)
print("=" * 100)

# Append to the experiments markdown (create if absent).
with open(md_path, "a", encoding="utf-8") as f:
    f.write(block + "\n")
print(f"Appended experiment record to: {md_path}")
PYEOF
    
    log_info "Benchmark log:    $BENCHMARK_LOG"
    log_info "Results table:    $RESULTS_TSV"
    log_info "Experiments doc:  $EXPERIMENTS_MD"
    log_info "Script execution complete!"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log_info "Starting gpt-oss-20b CUDA Model Build, Benchmark & MMLU"
    log_info "Script directory: $SCRIPT_DIR"
    echo ""
    
    step_venv_preparation
    install_ort_from_source
    setup_environment
    ensure_genai_from_source
    build_variants
    run_benchmarks
    run_mmlu
    generate_summary
    
    log_info "All steps completed successfully!"
    echo ""
}

# Execute main function
main "$@"
