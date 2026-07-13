# GPT-OSS-20B Quantization Experiments (CUDA EP)

Consolidated record of the int4/int8 quantization experiments run on `openai/gpt-oss-20b`
for the ONNX Runtime CUDA execution provider, including each model's Olive recipe, on-disk
path, configuration, and measured results.

- **Hardware**: 8×H200 (sm_90), CUDA 13.0
- **Stack**: onnxruntime 1.28.0, onnxruntime-genai 0.14.0-dev0, Olive (ModelBuilder pass)
- **Model dims**: hidden `K=2880`, vocab `N=201088`, 24 layers, MoE 32 experts,
  `tie_word_embeddings=False` (separate `lm_head`). Body = 72 `MatMulNBits` nodes;
  MoE experts are separate `QMoE` ops.
- **MMLU**: `match_mmlu`, full set = 14042 samples unless marked *(800-sample)*.
- **Decode TPS**: `benchmark_e2e.py`, GPU0, batch 1, prompt 512, gen 128, 5 runs / 2 warmup.

> All recipe paths below are relative to `gpt-oss-20b/`. All model paths are the Olive
> output directories produced by running the corresponding recipe.

---

## 1. Configuration dimensions

Every model is described by five independent configuration axes.

### 1.1 Embedding (`model.embed_tokens`)
- Controlled by whether `Gather` is included in `int4_op_types_to_quantize`.
- `['MatMul', 'Gather']` → embedding quantized to **int4** (`embed_tokens.weight_Q4` + `weight_scales`).
- `['MatMul']` → embedding kept **FP16** (`embed_tokens.weight`).
- Quantizing the embedding saves disk but is on the input side only (no compute on hot path).

### 1.2 lm_head (separate output projection, `K=2880`, `N=201088`)
- `default` int4 → `lm_head.MatMul.weight_Q4` + `weight_scales` (MLAS `DefaultWeightOnlyQuantizer`, scale = max/8).
- `rtn` int4 → `weight_Q4G32` + `weight_scale` (RTN, scale = max/7.5).
- **int8** (`last_matmul_weight_int8: true`) → `weight_Q8G32` (uint8, offset-128) + `weight_scale`.
  - The int8 pass **must** use `RTNWeightOnlyQuantConfig`. The MLAS `default` 8-bit path emits
    *signed* int8 with *negative* scales, which is incompatible with the `MatMulNBits` int8
    runtime kernel (kernel expects unsigned uint8 offset-by-128 with positive scales).
    This was the bug fixed in `base.py::to_int4` (the broken build scored 0.6722, see §3).

### 1.3 Body (72 attention/router `MatMulNBits`, block size 32)
Selected via `int4_algo_config` / `base_method`:
- `default` — MLAS `DefaultWeightOnlyQuantizer` (scale = max/8).
- `rtn` — `RTNWeightOnlyQuantConfig` (scale = max/7.5).
- `k_quant` — `KQuantWeightOnlyQuantConfig`.
- The body is the dominant accuracy driver (see Conclusions).

### 1.4 Mixed bit-placement
Per-layer int8 placement flags (implemented, evaluated only for `*_last_matmul_weight_int8`):
- `last_matmul_weight_int8` — promote only `lm_head` to int8 (used by the lmh8 models).
- `int8_mixed_layers` — llama.cpp-style: first/last eighth + every 3rd layer at 8 bits *(not benchmarked)*.
- `int8_linear_attn` — int8 for linear-attention projections *(not benchmarked)*.
- Legacy `int4_algo_config` aliases: `rtn_last`, `k_quant_last`, `k_quant_mixed`, `k_quant_linear`.
- QMoE block size (`qmoe_block_size`) is a separate knob: 32 or 64 (128 invalid — `K=2880` not divisible by 128).

### 1.5 CUDA graph
- `enable_cuda_graph` is **disabled (`"0"`)** in `genai_config.json` for every model in this study.
- No CUDA-graph A/B benchmark was performed; decode TPS numbers below are eager (graph off).
- `enable_skip_layer_norm_strict_mode: "1"` is set on the fixed default+int8 model.

---

## 2. Models, recipes, paths, and results

| # | Model dir (`gpt-oss-20b/cuda/…`) | Olive recipe (`gpt-oss-20b/cuda/…`) | Embedding | Body | lm_head | QMoE blk | Size | Decode TPS | MMLU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `variants/cuda_int4_int4_qmoe_default` | `gpt-oss-20b_cuda_int4_int4_qmoe_default.json` | int4 | default int4 | default int4 | 32 | 11.011 GiB | — | 0.7163 *(800-sample)* |
| 2 | `variants/cuda_int4_int4_qmoe_rtn_matmul_only` | `gpt-oss-20b_cuda_int4_int4_qmoe_rtn_matmul_only.json` | FP16 | rtn int4 | rtn int4 | 32 | ~11.76 GiB | — | 0.7762 |
| 3 | `variants/cuda_int4_int4_qmoe_rtn_last_matmul_only` | `gpt-oss-20b_cuda_int4_int4_qmoe_rtn_last_matmul_only.json` | FP16 | rtn int4 | **int8** (rtn) | 32 | ~11.8 GiB | — | 0.7769 |
| 4 | `variants/cuda_int4_int4_qmoe_k_quant_mixed` | `gpt-oss-20b_cuda_int4_int4_qmoe_k_quant_mixed.json` | int4 | k_quant mixed | k_quant | 32 | — | — | not evaluated |
| 5 | `variants/cuda_int4_int8_qmoe_default` | `gpt-oss-20b_cuda_int4_int8_qmoe_default.json` | int4 | default int4 | default int4 | int8 MoE | — | — | not evaluated |
| 6 | `variants/cuda_int4_int8_qmoe_k_quant_mixed` | `gpt-oss-20b_cuda_int4_int8_qmoe_k_quant_mixed.json` | int4 | k_quant mixed | k_quant | int8 MoE | — | — | not evaluated |
| 7 | `variants/cuda_int4_rtn_matmul_only_lmh8_bs32` | `gpt-oss-20b_cuda_int4_rtn_matmul_only_lmh8_bs32.json` | FP16 | rtn int4 | **int8** (rtn) | 32 | 12.056 GiB | 261.78 | 0.7763 |
| 8 | `variants/cuda_int4_rtn_matmul_only_lmh8_bs64` | `gpt-oss-20b_cuda_int4_rtn_matmul_only_lmh8_bs64.json` | FP16 | rtn int4 | **int8** (rtn) | 64 | 11.499 GiB | 261.34 | 0.7770 |
| 9 | `model_default_lmh8_v2` | `tmp_default_lmh8.json` *(temp, not in info.yaml)* | int4 | default int4 | **int8** (rtn, fixed) | 32 | 11.281 GiB | — | **0.6739** *(full)* |
| 10 | `model_rtn_mixed_lmh8_bs64` | `gpt-oss-20b_cuda_int4_rtn_mixed_lmh8_bs64.json` | FP16 | rtn int4 + **mixed int8** | **int8** (rtn) | 64 | 11.581 GiB | 256.36 | **0.8010** *(full)* |
| R | Foundry reference (external build, no repo recipe) | — | — | per-channel MoE | int8 | — | 11.038 GiB | 260.03 *(cg=1)* | 0.7929 |

Models 1–8 use `output_dir: model` in their recipe; the table lists the renamed `variants/…`
directories where the built models are archived. Model 9 uses `output_dir: model_default_lmh8_v2`;
model 10 uses `output_dir: model_rtn_mixed_lmh8_bs64`.

### 2.1 Recipe `extra_options` quick reference
- #1: `{qmoe_block_size: 32}`, `int4_op_types_to_quantize: [MatMul, Gather]`
- #2/#3: `{qmoe_block_size: 32}`, `int4_op_types_to_quantize: [MatMul]`, #3 adds `int4_algo_config: rtn_last`
- #5: `{use_8bits_moe: true, qmoe_block_size: 32}`, `[MatMul, Gather]`
- #7: `{int4_block_size: 32, qmoe_block_size: 32}`, `int4_algo_config: rtn_last`, `[MatMul]`
- #8: `{int4_block_size: 32, qmoe_block_size: 64}`, `int4_algo_config: rtn_last`, `[MatMul]`
- #9: `{qmoe_block_size: 32, last_matmul_weight_int8: true}`, `[MatMul, Gather]`
- #10: `{int4_block_size: 32, qmoe_block_size: 64, last_matmul_weight_int8: true, int8_mixed_layers: true}`,
  `int4_algo_config: rtn`, `[MatMul]` — rtn body with llama.cpp-style int8 mixed layers + int8 lm_head, 4-bit QMoE block 64

### 2.2 ONNX graph node-count summary

> **Source:** these counts are read **directly from the built `model.onnx` graphs**
> (loaded with the `onnx` Python API and counted by `node.op_type` / attributes), not derived
> from theory. `has-zero-point` for `MatMulNBits` = the 4th input (`zero_points`) is present and
> non-empty; for `QMoE` = a zero-point input is present. `4-bit` = `bits == 4` (`MatMulNBits`) /
> `expert_weight_bits == 4` (`QMoE`).

Columns: `GBQ` = `GatherBlockQuantized` (quantized embedding), `GQA` = `GroupQueryAttention`,
`MatMulNBits` total / 4-bit / has-zero-point, `QMoE` total / 4-bit / has-zero-point.

| Model | GBQ | GQA | MatMulNBits (tot / 4-bit / zp) | QMoE (tot / 4-bit / zp) |
|---|---|---|---|---|
| `cuda_int4_int4_qmoe_default` | 1 | 24 | 73 / 73 / 0 | 24 / 24 / 0 |
| `cuda_int4_int4_qmoe_rtn_matmul_only` | 0 | 24 | 73 / 73 / 0 | 24 / 24 / 0 |
| `cuda_int4_int4_qmoe_rtn_last_matmul_only` | 0 | 24 | 73 / 72 / 0 | 24 / 24 / 0 |
| `cuda_int4_int4_qmoe_k_quant_mixed` | 0 | 24 | 73 / 60 / 73 | 24 / 24 / 0 |
| `cuda_int4_int8_qmoe_default` | 1 | 24 | 73 / 73 / 0 | 24 / 0 / 0 |
| `cuda_int4_int8_qmoe_k_quant_mixed` | 0 | 24 | 73 / 60 / 73 | 24 / 0 / 0 |
| `cuda_int4_rtn_matmul_only_lmh8_bs32` | 0 | 24 | 73 / 72 / 0 | 24 / 24 / 0 |
| `cuda_int4_rtn_matmul_only_lmh8_bs64` | 0 | 24 | 73 / 72 / 0 | 24 / 24 / 0 |
| `model_default_lmh8_v2` | 1 | 24 | 73 / 72 / 0 | 24 / 24 / 0 |
| `model_rtn_mixed_lmh8_bs64` | 0 | 24 | 73 / 60 / 0 | 24 / 24 / 0 |
| `foundry_cuda_v1` (reference) | 0 | 24 | 73 / 60 / **73** | 24 / 24 / 0 |

Reading the table:
- **GBQ = 1** exactly when the embedding is quantized to int4 (recipe includes `Gather`);
  `0` means FP16 embedding.
- **GQA = 24** (one per decoder layer) and **MatMulNBits total = 73** (72 body + 1 `lm_head`)
  for every model — the topology is identical; only weight precision changes.
- **MatMulNBits 4-bit count**: `73` when `lm_head` is also 4-bit; `72` when `lm_head` is promoted
  to int8 (`rtn_last` / lmh8 / fixed default+int8). `k_quant_mixed`, `model_rtn_mixed_lmh8_bs64`
  and `foundry_cuda_v1` show `60`, i.e. 13 of the 73 `MatMulNBits` are at 8-bit (12 mixed `qkv_proj`
  layers + the int8 `lm_head`).
- **MatMulNBits has-zero-point**: `73` for the `k_quant` models **and for `foundry_cuda_v1`** —
  these use **asymmetric** quantization (every `MatMulNBits` carries a `zero_points` input).
  All `rtn`/`default` algos are symmetric (`0` zero-points) — including `model_rtn_mixed_lmh8_bs64`,
  which has the same 13 int8 layers as `k_quant_mixed` but **without** zero-points (rtn is symmetric).
- **Foundry asymmetric attention**: because `foundry_cuda_v1` is asymmetric, each `qkv_proj` /
  `o_proj` `MatMulNBits` holds **both** a packed-weight UINT8 tensor **and** a `zero_points`
  UINT8 tensor — so the UINT8 initializer count over attention projections is **≈ ×48**
  (2 per node × 24 layers) versus **×24** for the symmetric `rtn` models (weight only). The node
  *count* is identical (73); the difference is the extra per-node zero-point tensors, reflected in
  the `zp = 73` column.
- **QMoE 4-bit count**: `24` for int4 experts; `0` when `use_8bits_moe: true` (int8 experts).
- **QMoE has-zero-point**: `0` everywhere — MoE experts are quantized symmetrically.

---

## 3. Root-cause analysis: default body is the accuracy limiter

A two-pass `default` body + int8 `lm_head` build (model #9) initially scored **0.6722** because the
int8 pass used the MLAS `default` 8-bit format (signed int8 / negative scales), which the
`MatMulNBits` int8 kernel cannot consume. After fixing `to_int4` to use
`RTNWeightOnlyQuantConfig` for the int8 pass:

- The rebuilt `lm_head` became byte-identical to a known-good rtn-int8 lm_head
  (uint8 mean 127.9, positive scales 2.0e-6 … 6.4e-4).
- The body remained byte-identical to the plain `default` body (437 common inits, 0 differing;
  72 body `MatMulNBits` identical).
- The fixed model still scored only **0.6739** on the full 14042-sample MMLU.

**Verified full-eval ladder** (saved `summary_report.md`):

| Body / lm_head algo | MMLU (full) |
|---|---|
| Foundry reference (int8 lm_head) | 0.7929 |
| rtn int4 / int8 lm_head (bs64) | 0.7770 |
| rtn int4 / int8 lm_head (rtn_last) | 0.7769 |
| rtn int4 / int8 lm_head (bs32) | 0.7763 |
| rtn int4 / int4 lm_head | 0.7762 |
| default int4 / int4 lm_head | 0.7163 *(800-sample; no full eval)* |
| default int4 / **int8** lm_head (fixed #9) | **0.6739** |
| default int4 / int8 lm_head (broken format) | 0.6722 |

The earlier "default = 0.7980" premise was a **mis-attribution**: no saved full eval shows 0.798
for any `default`-body model. The 0.7980 figure belongs to an **rtn** variant (`rtn_matmul_only`).
The measured `default` body is ~0.716, well below rtn ~0.777.

---

## 4. Performance

### 4.1 lmh8 decode sweep (CUDA graph off)
- Decode throughput is essentially flat between the two lmh8 builds:
  bs32 = **261.78 tps**, bs64 = **261.34 tps** (within noise).
- Lowering `qmoe_block_size` from 32 → 64 reduces size by ~0.56 GiB
  (12.056 → 11.499 GiB) with flat MMLU (0.7763 → 0.7770) and flat decode → **block 64 strictly better**.

### 4.2 Top-variant accuracy + performance (merged)

Sizes recomputed consistently in **GiB** (÷2³⁰) from the actual `model.onnx.data` files.
Prefill/decode from `benchmark_e2e.py`.

| Model | Full MMLU | Size (GiB) | Prefill (tps) | Decode (tps) | Decode (ms/tok) | cuda_graph |
|---|---:|---:|---:|---:|---:|:--:|
| `rtn_matmul_only` | 0.7980 | 11.76 | 20864.5 | **297.7** | 3.36 | 1 |
| `k_quant_mixed` | **0.8085** | 12.14 | 20150.6 | 270.7 | 3.69 | 1 |
| `rtn_mixed_lmh8_bs64` (new) | 0.8010 | 11.58 | 20348.5 | 256.4 | 3.90 | 1 |
| `foundry_cuda_v1` (cg=1) | 0.7929 | **11.04** | 23850.3 | 260.0 | 3.85 | **1** |
| `foundry_cuda_v1` (cg=0) | 0.7929 | **11.04** | **25845.5** | 236.1 | 4.24 | 0 |

- `rtn_matmul_only` = `variants/cuda_int4_int4_qmoe_rtn_matmul_only`.
- `k_quant_mixed` = `variants/cuda_int4_int4_qmoe_k_quant_mixed`
  (`/tianlei/models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_k_quant_mixed/`).
- `rtn_mixed_lmh8_bs64` = `cuda/model_rtn_mixed_lmh8_bs64` (recipe
  `gpt-oss-20b_cuda_int4_rtn_mixed_lmh8_bs64.json`): rtn body + int8 mixed layers + int8 lm_head,
  4-bit QMoE block 64. **0.8010** full MMLU — essentially matching `k_quant_mixed` (0.8085) but
  ~0.56 GiB smaller (symmetric, no zero-points) and with FP16 embedding.
- **Foundry CUDA-graph A/B:** enabling `enable_cuda_graph=1` lifts Foundry decode from **236.1 → 260.0 tps**
  (+10%), at a small prefill cost (25845 → 23850 tps, graph capture overhead on the prefill path).
  Even with cg=1, Foundry decode (260.0) trails `rtn_matmul_only` (297.7).
- **Size-unit correction:** an earlier quote of `foundry_cuda_v1` ≈ 11.85 GB used decimal GB
  (÷10⁹); in GiB (÷2³⁰, the unit used for the other rows) it is **11.04 GiB** — actually the
  *smallest* of the set, not the largest.
- All decode rows above except the explicit `foundry_cuda_v1 (cg=0)` row were measured with cg=1
  (batch 1, prompt 512, gen 128, 5 reps / 2 warmup, GPU pinned).

### 4.3 Why Foundry is the smallest despite asymmetric attention + int8 lm_head

The size advantage comes almost entirely from **per-channel** (vs **block-wise**) MoE scales:

| Tensor | `rtn_matmul_only` | `foundry_cuda_v1` |
|---|---|---|
| `gate_up_proj.scales` | FP16 `[32, 5760, 90]` (block_size=32) | FP16 `[32, 5760]` (per-channel) |
| `down_proj.scales` | FP16 `[32, 2880, 90]` (block_size=32) | FP16 `[32, 2880]` (per-channel) |
| MoE scale bytes (24 layers) | **~1.15 GiB** (768 + 384 MiB) | **~25 MiB** (17 + 8 MiB) |

rtn uses **block-wise** quantization (one scale per 32 input elements → the extra `90`
dimension), while Foundry uses **per-channel** quantization (a single scale per output row),
which removes the 90× block dimension and drops MoE scales by **~1.13 GiB**.

**Net size math** (`foundry_cuda_v1` vs `rtn_matmul_only`):
- **−1.13 GiB** from coarse per-channel MoE scales
- **+~0.4 GiB** spent back: int8 `lm_head` (+294 MiB vs rtn's int4) and asymmetric attention
  (extra zero-point UINT8 tensors — qkv/o UINT8 tensor count is ×48 in Foundry vs ×24 in rtn)
- ≈ **−0.72 GiB** → **11.04 GiB vs 11.76 GiB** ✓

### 4.4 Comprehensive decode/prefill sweep across sequence lengths (CUDA graph ON)

Batch 1, gen 128, 5 reps / 2 warmup, GPU-pinned (`CUDA_VISIBLE_DEVICES=1`), all with
`enable_cuda_graph=1`. Three models: `rtn_matmul_only` (all-int4), `rtn_mixed_lmh8_bs64`
(13 int8 layers + int8 lm_head + QMoE block 64), `foundry_cuda_v1` (per-channel MoE, int8 lm_head).

**Decode throughput (tps):**

| Model | l=256 | l=512 | l=1024 | l=2048 |
|---|---:|---:|---:|---:|
| `rtn_matmul_only` | **297.3** | **297.1** | **284.5** | **275.9** |
| `rtn_mixed_lmh8_bs64` | 257.0 | 257.1 | 248.0 | 241.1 |
| `rtn_mixed_lmh8_bs64` *(strict_mode=0 fix)* | 277.2 | 275.6 | 266.4 | 258.5 |
| `foundry_cuda_v1` | 259.2 | 259.8 | 250.4 | 244.5 |

**Prefill throughput (tps):**

| Model | l=256 | l=512 | l=1024 | l=2048 |
|---|---:|---:|---:|---:|
| `rtn_matmul_only` | 15467 | 20900 | 26582 | 30235 |
| `rtn_mixed_lmh8_bs64` | 14928 | 20503 | 25647 | 29970 |
| `foundry_cuda_v1` | **17813** | **23910** | **30115** | **32408** |

- `rtn_matmul_only` leads decode at every length; `rtn_mixed` and `foundry` are nearly identical
  (~257 vs ~259) because both carry an int8 lm_head + int8/wider attention path.
- Decode degrades gently with longer KV (≈ −7% from 256 → 2048 for all models) — KV-cache reads,
  not weights, are the length-dependent term.
- Foundry wins prefill (more compute-efficient per-channel MoE) but not decode.

### 4.5 Root cause of the `rtn_matmul_only` (297) vs `rtn_mixed` (257) decode gap

The ~15% / ~0.53 ms-per-token decode gap was attributed with **nsys** kernel profiling
(CUDA graph **off** for per-kernel visibility, GPU 1). The QMoE kernels are **byte-for-byte
identical** between the two models — `qmoe_block_size` 32 vs 64 has **zero** decode cost. The gap
decomposes into three contributors:

| Contributor | Δ µs/tok (B−A) | Share | Cause |
|---|---:|---:|---|
| **`enable_skip_layer_norm_strict_mode=1`** | **+277** | **52%** | disables the fused `SkipLayerNormKernelSmall`, falling back to a ~3.6× slower standalone `cuApplyLayerNorm` (48 norms/token) |
| **int8 lm_head vs int4** | +187 | 35% | reads 2× weight bytes (579 vs 290 MB) on N=201088 |
| **12 int8 body MatMulNBits vs int4** | +73 | 14% | 2× weight bytes on the promoted `qkv_proj` layers |
| QMoE (block 32 → 64) | ~0 | 0% | identical kernels |
| attention / other | +6 | ~1% | |
| **Sum** | **+537** | | (measured delta = 536 µs nsys / 524 µs graph-ON) |

**Key insight — the largest contributor is a config flag, not int8 precision.** The int8
`MatMulNBits`/lm_head kernels are **not** inherently less bandwidth-efficient than int4 — both run
at ~1.56 TB/s (~0.64 µs/MB); int8 is slower only because it moves 2× the bytes. The dominant
51% comes from `enable_skip_layer_norm_strict_mode=1` (set on `rtn_mixed`, off on `rtn_matmul_only`),
which forces a slow non-fused LayerNorm path.

**Validated fix:** setting `enable_skip_layer_norm_strict_mode=0` on `rtn_mixed_lmh8_bs64` recovers
**~7% decode at every length** (257 → 277, 248 → 266, 241 → 258 tps; see §4.4 row), closing roughly
half the gap to `rtn_matmul_only` with no graph/weight changes. The remaining ~20 tps is the
intrinsic int8 lm_head + int8 body cost (the accuracy/throughput price of the +0.024 MMLU gain).
The model's evaluated `genai_config.json` was left at its original `strict_mode=1` (the MMLU 0.8010
number was measured with it on); flip the flag to trade a small numerical-precision margin for ~7%
decode.

Profiling traces: `/tmp/nsys_A_rtn_matmul_only.nsys-rep`, `/tmp/nsys_B_rtn_mixed_lmh8_bs64.nsys-rep`.

---

## 5. Cross-Engine Throughput & MMLU Evaluation (ORT vs llama.cpp vs vLLM)

### 5.1 Setup & Fairness Constraint

A three-engine comparative evaluation was conducted across ORT (`cuda_int4_int4_qmoe_k_quant_mixed`
and `cuda_int4_int8_qmoe_k_quant_mixed`), llama.cpp (MXFP4), and vLLM (bfloat16 native) to assess
relative accuracy and throughput in a fair apples-to-apples setting. **Fairness requirement**: all engines
evaluated with reasoning enabled to match ORT's reasoning-medium capability. As noted in the initial
specification: *"ORT is evaluated with reasoning medium. If we turn off reasoning for llama.cpp, that
might not be a fair comparison."*

**Completion adapter**: a custom `OpenAIChatToCompletionLetterFn` adapter (evals library) with
chat-completions API, configured with `max_tokens: 768, temperature: 0.0, top_p: 1.0`. This adapter:
- Extracts final answer letters (A/B/C/D) from model responses via fallback chain: explicit
  "answer:" markers → bracketed format `[X]` → last standalone token
- For reasoning-mode responses, inspects both `message.content` and `message.reasoning_content` fields
- Calibrated token budget of 768 was empirically determined to yield ~94% answer-extraction success
  for llama.cpp with reasoning enabled (vs. ~10% at 256 tokens)

### 5.2 Throughput across sequence lengths (batch=1, gen=128, cuda_graph=1)

| Engine | Seq Len | Prefill (tps) | Decode (tps) | Decode (ms/tok) | Model / Config |
|--------|--------:|----------:|----------:|----------:|---|
| **ORT int4 k_quant** | 256 | 15,150 | 269 | 3.71 | `cuda_int4_int4_qmoe_k_quant_mixed` |
| | 512 | 20,002 | 269 | 3.71 | |
| | 1024 | 26,870 | 259 | 3.85 | |
| | 2048 | 29,630 | 251 | 3.98 | |
| **ORT int8 k_quant** | 256 | 14,788 | 264 | 3.78 | `cuda_int4_int8_qmoe_k_quant_mixed` |
| | 512 | 19,583 | 264 | 3.78 | |
| | 1024 | 26,791 | 254 | 3.93 | |
| | 2048 | 30,166 | 248 | 4.03 | |
| **llama.cpp** (MXFP4) | 256 | 6,652 | 296 | 3.37 | `--reasoning on` |
| | 512 | 9,551 | 297 | 3.36 | |
| | 1024 | 9,712 | 297 | 3.37 | |
| | 2048 | 9,780 | 297 | 3.36 | |
| **vLLM** (fp16) | 256 | — | 279* | 3.58 | bfloat16 dtype, e2e latency |
| | 512 | — | 277* | 3.61 | |
| | 1024 | — | 272* | 3.68 | |
| | 2048 | — | 263* | 3.80 | |

**Key observations**:
- **Decode throughput leader**: llama.cpp achieves stable **296–297 tps** across all sequence lengths,
  outperforming ORT int4 (269 tps @ 256–512) and vLLM (~263–279 tps).
- **Prefill**: ORT int4 prefill dominates at longer sequences (30k tps @ 2048 tokens), while llama
  plateaus at ~9.8k tps. vLLM prefill omitted (e2e measurement only).
- **Sequence-length robustness**: ORT and vLLM decode throughput degrades ~5–7% from 256 → 2048 (KV-cache
  reads); llama.cpp is flat (cache-efficient or different kernel characteristics).
- **ORT int4 vs int8 k_quant**: negligible throughput difference (~1–3%), confirming the k_quant body
  (with zero-points) does not impose a per-bit-width penalty. The QMoE int8/int4 distinction is in
  tensor precision, not kernel specialization.

### 5.3 MMLU Evaluation (14,042 samples, 8-GPU sharding)

**Finalized llama.cpp result**:
- **llama.cpp** (reasoning=on, mt=768): **0.8252** full MMLU (11587/14042)
  from `/tmp/llama_mmlu_full_reasoning_on_chatadapter_mt768_final_20260619_021049/summary_report.md`.

**vLLM status**:
- Full vLLM MMLU was blocked in the original environment by startup failures.
- Root causes identified and fixed in an isolated environment (see §5.4); smoke validation now passes.

**Smoke tests** validate adapter and configuration:

| Engine | Test Type | Samples | Accuracy | Config |
|--------|-----------|--------:|----------:|---|
| llama.cpp | Smoke | 100 | 0.82 | mt=384, reasoning on |
| vLLM | Smoke | 100 | 0.91 | bfloat16, mt=768 |

**Observed outcomes (smoke, 100 samples):**
- llama.cpp mt=768: **0.93**
- vLLM mt=768: **0.91** (after environment fix)
- ORT mt=768 (pre-fix): **0.91**
- ORT mt=768 (after ORT letter-extraction patch): **0.93**
- ORT mt=2048: **0.94**

These numbers show strong mt sensitivity on ORT and indicate that comparing ORT at `max_new_tokens=2048`
versus llama/vLLM at `max_tokens=768` is not apples-to-apples.

**Completion function parity**: both llama.cpp (port 8081) and vLLM (port 8000) registered in
`~/.evals/completion_fns/gpt_oss_local.yaml` with identical adapter (`OpenAIChatToCompletionLetterFn`)
and generation parameters (`max_tokens: 768, temperature: 0.0, top_p: 1.0`). This ensures any accuracy
differences reflect model/quantization effects, not evaluation harness differences.

### 5.4 Why ORT can look lower (80 vs 82) and how to close the gap

The apparent ORT gap is not explained by one factor; the evidence points to two contributors:

1. **Token-budget mismatch (`mt`)**
   - ORT completion fn (`oss/gpt-oss-20b`) defaults to `max_new_tokens: 2048`.
   - llama/vLLM evals were configured with `max_tokens: 768`.
   - At matched `mt=768` (100-sample smoke), ORT=0.91 and llama=0.93 (gap ≈2 points).
   - At `mt=2048`, ORT improves to 0.94 on the same smoke set.

2. **Answer-format extraction mismatch in ORT path (now fixed)**
   - ORT completion fn previously extracted the `final` channel text but did not enforce A/B/C/D output.
   - In mt=768 smoke before the patch, **5/100** ORT samples returned non-letter free-form text.
   - After adding the same letter fallback chain used by `OpenAIChatToCompletionLetterFn`, ORT mt=768
     improved from **0.91 → 0.93** on a fresh 100-sample rerun, with **0/100 non-letter outputs**.
   - Example misses: `match_mmlu_shard_2.local.1`, `match_mmlu_shard_5.local.0`, where ORT produced
     reasoning text and no single-letter answer while llama produced correct letters.

**Practical gap-closing plan**:
- **For fair engine comparison**: run all engines with the same token budget (`mt=768` or `mt=2048`).
- **For ORT eval quality**: keep the added A/B/C/D letter-extraction fallback in
  `evals/completion_fns/ort_genai.py` enabled (`extract_letter_choice=true`).
- **For best ORT full accuracy**: keep higher `max_new_tokens` (>=2048) when reasoning is enabled,
  or verify that lowering to 768 does not truncate final-answer emission for the chosen model variant.
- **For model-side uplift**: use top ORT quantization variants (`k_quant_mixed` / `rtn_mixed_lmh8_bs64`)
  and keep `enable_skip_layer_norm_strict_mode=0` for throughput while validating accuracy deltas.

---

## 6. Conclusions

1. **Use `rtn` (or `k_quant`) — not `default` — for int8-lm_head models.** The MLAS `default`
   int4 body is the accuracy limiter (~0.67–0.72), independent of the lm_head precision.
2. **The int8 lm_head is roughly neutral** (rtn int4 / int4 = 0.7762 vs rtn int4 / int8 = 0.7769,
   ≈ +0.0007). It is not the source of any regression.
3. **The int8 pass must use RTN.** The MLAS `default` 8-bit QOperator format (signed int8 +
   negative scales) is incompatible with the `MatMulNBits` int8 kernel; this caused the 0.6722 build.
4. **Best practical models**: `k_quant_mixed` (0.8085) and the new **`rtn_mixed_lmh8_bs64`**
   (0.8010) are the **top accuracy** tier; `rtn_mixed_lmh8_bs64` nearly matches k_quant accuracy
   while being **~0.56 GiB smaller** (11.58 vs 12.14 GiB) thanks to symmetric quantization (no
   zero-points) and 4-bit QMoE block 64. `rtn_matmul_only` gives the **best decode**
   (0.7980 MMLU, 297.7 tps, 11.76 GiB); the Foundry reference is the **smallest**
   (0.7929, 11.04 GiB) and best prefill.
5. **Mixed int8 layers help accuracy a lot for rtn**: promoting the 12 most-sensitive `qkv_proj`
   layers (+ int8 lm_head) lifts rtn from 0.7770 (plain rtn lmh8 bs64) to **0.8010** (+0.024) at
   only +0.08 GiB, while decode stays ~256 tps. This is the single best accuracy/size trade in the set.
6. **Foundry is smallest because of per-channel MoE scales** (~1.13 GiB saved vs block-wise rtn),
   which more than pays for its int8 lm_head + asymmetric attention overhead (~0.4 GiB).
7. **Prefer `qmoe_block_size: 64`** over 32 — smaller with no accuracy/throughput cost.
   `qmoe_block_size: 128` is invalid for this model (`K=2880` not divisible by 128).
8. **CUDA graph helps decode ~10%**: Foundry decode 236.1 → 260.0 tps with `enable_cuda_graph=1`
   (small prefill cost). Even so, `rtn_matmul_only` (297.7) remains the decode leader.
9. **The `rtn_matmul_only` vs `rtn_mixed` decode gap is ~half a config flag, not int8.** nsys
   attribution (§4.5): `enable_skip_layer_norm_strict_mode=1` on `rtn_mixed` costs **+277 µs/tok
   (52% of the gap)** by disabling the fused `SkipLayerNormKernelSmall`; int8 lm_head (+187 µs) and
   12 int8 body layers (+73 µs) are the rest; `qmoe_block_size` 32→64 costs nothing. Setting
   `enable_skip_layer_norm_strict_mode=0` recovers **~7% decode** (257 → 277 tps) with no weight or
   graph change. The int8 `MatMulNBits` kernel is **not** less bandwidth-efficient than int4 — both
   run at ~1.56 TB/s; int8 is slower only because it moves 2× the bytes.
10. **Decode scales gently with sequence length** (§4.4): ≈ −7% from l=256 → l=2048 for all three
    models (KV-cache reads, not weights). `rtn_matmul_only` leads decode at every length; Foundry
    leads prefill at every length.

---

## 7. Reproduction notes

- Build: `olive run --config <recipe>.json` (uses the installed `onnxruntime_genai` package).
- MMLU eval must be launched from `/tmp` (not `/home/tianlei`) to avoid the local `evals/`
  directory shadowing the installed `evals` package.
- Eval harness: `/tianlei/scripts/h200_18/run_evals_parallel.sh --shard match_mmlu --parallel 1 --gpus "0..7"`;
  accuracy from `OUTDIR/mmlu/summary_report.md` ("Combined result").

### 7.1 Cross-engine MMLU evaluation setup

For fair llama.cpp and vLLM comparison (§5):
1. **llama.cpp server**: `llama-server --reasoning on --max-tokens 768` (port 8081)
2. **vLLM server**: `vllm serve --dtype bfloat16 --max-model-len 131072` (port 8000)
3. **Completion function**: both registered in `~/.evals/completion_fns/gpt_oss_local.yaml` to
   `OpenAIChatToCompletionLetterFn` with `max_tokens: 768, temperature: 0.0, top_p: 1.0`
4. **Sharding**: 8 GPUs × 1756–1755 samples/shard (14,042 total), parallel launch
5. **Adapter**: uses chat-completions API with fallback extraction from `content` and `reasoning_content` fields

## Experiment (2026-06-21): rtn qk-norm-fusion variants (v2 build script)

- **Generated:** 2026-06-21T06:49:56
- **Build:** Olive recipes via `cuda/run_gpt_oss_model_build_v2.sh`, onnxruntime-genai **built from source** (patched model builder).
- **Decode TPS:** `benchmark_e2e.py`, batch 1, prompt 512, gen 128, CUDA graph=1, **XQA=1**.
- **MMLU:** `match_mmlu`, full (14042) samples, multi-GPU shard pooled accuracy.

| Model (variant) | Size (GiB) | Prefill TPS | Decode TPS | MMLU | Notes |
|---|---:|---:|---:|---:|---|
| `cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs0` | 10.68 | 29612.3 | 390.8 | 0.7634 | rtn int4 body (blk32), int4 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion |
| `cuda_int4_int4_qmoe_rtn_last_matmul_only_qknorm_bs0` | 10.95 | 28936.8 | 369.8 | 0.7598 | rtn int4 body (blk32), int8 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion |
| `cuda_int4_int4_qmoe_rtn_matmul_only_qknorm_bs64` | 10.65 | 29267.5 | 390.9 | 0.7470 | rtn int4 body (blk64), int4 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion |
| `cuda_int4_int4_qmoe_rtn_mixed_matmul_only_qknorm_bs0` | 11.04 | 28608.8 | 362.9 | 0.7933 | rtn int4 body + mixed int8 layers, int8 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion |


## Experiment (2026-06-21): rtn qk-norm-fusion variants (v2 build script)

- **Generated:** 2026-06-21T16:29:50
- **Build:** Olive recipes via `cuda/run_gpt_oss_model_build_v2.sh`, onnxruntime-genai **built from source** (patched model builder).
- **Decode TPS:** `benchmark_e2e.py`, batch 1, prompt 512, gen 128, CUDA graph=1, **XQA=1**.
- **MMLU:** `match_mmlu`, full (14042) samples, multi-GPU shard pooled accuracy.

| Model (variant) | Size (GiB) | Prefill TPS | Decode TPS | MMLU | Notes |
|---|---:|---:|---:|---:|---|
| `cuda_int4_int4_qmoe_rtn_mixed_lmh4_qknorm_qmoe0` | 10.77 | 29403.1 | 382.0 | 0.8017 | A: rtn int4 body (blk32) + mixed int8 layers, int4 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion |
| `cuda_int4_int4_qmoe_rtn_matmul_only_lmh4_qknorm_qmoe64` | 11.23 | 25487.3 | 385.8 | 0.7901 | B: rtn int4 body (blk32), int4 lm_head, FP16 embed, QMoE block64, qk-norm fusion |
| `cuda_int4_int4_qmoe_rtn_mixed_lmh4_qknorm_qmoe64` | 11.31 | 25442.9 | 377.7 | 0.8087 | C: rtn int4 body (blk32) + mixed int8 layers, int4 lm_head, FP16 embed, QMoE block64, qk-norm fusion |
| `cuda_int4_int4_qmoe_kquant_mixed_lmh4_qknorm_qmoe64` | 11.33 | 25664.2 | 363.7 | 0.8156 | E: k_quant int4 body (blk32) + mixed int8 layers, int4 lm_head, FP16 embed, QMoE block64, qk-norm fusion |


## Experiment (2026-06-21): rtn qk-norm-fusion variants (v2 build script)

- **Generated:** 2026-06-21T21:40:54
- **Build:** Olive recipes via `cuda/run_gpt_oss_model_build_v2.sh`, onnxruntime-genai **built from source** (patched model builder).
- **Decode TPS:** `benchmark_e2e.py`, batch 1, prompt 512, gen 128, CUDA graph=1, **XQA=1**.
- **MMLU:** `match_mmlu`, full (14042) samples, multi-GPU shard pooled accuracy.

| Model (variant) | Size (GiB) | Prefill TPS | Decode TPS | MMLU | Notes |
|---|---:|---:|---:|---:|---|
| `cuda_int4_int4_qmoe_kquant_mixed_lmh4_qknorm_qmoe0` | 10.79 | 29018.5 | 366.7 | 0.8128 | F: k_quant int4 body (blk32) + mixed int8 layers, int4 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion |
| `cuda_int4_int4_qmoe_kquant_matmul_only_lmh4_qknorm_qmoe64` | 11.25 | 25037.2 | 369.0 | 0.8101 | H: k_quant int4 body (blk32), int4 lm_head, FP16 embed, QMoE block64, qk-norm fusion |
| `cuda_int4_int4_qmoe_rtn_mixed_lmh4_qknorm_qmoe32` | 11.87 | 24421.5 | 374.7 | 0.8091 | G: rtn int4 body (blk32) + mixed int8 layers, int4 lm_head, FP16 embed, QMoE block32, qk-norm fusion |


## Experiment (2026-06-22): rtn qk-norm-fusion variants (v2 build script)

- **Generated:** 2026-06-22T00:30:04
- **Build:** Olive recipes via `cuda/run_gpt_oss_model_build_v2.sh`, onnxruntime-genai **built from source** (patched model builder).
- **Decode TPS:** `benchmark_e2e.py`, batch 1, prompt 512, gen 128, CUDA graph=1, **XQA=1**.
- **MMLU:** `match_mmlu`, full (14042) samples, multi-GPU shard pooled accuracy.

| Model (variant) | Size (GiB) | Prefill TPS | Decode TPS | MMLU | Notes |
|---|---:|---:|---:|---:|---|
| `cuda_int4_int4_qmoe_kquant_mixed_lmh8_qknorm_qmoe0` | 11.06 | 28371.4 | 348.9 | 0.8093 | I: k_quant int4 body (blk32) + mixed int8 layers, int8 lm_head, FP16 embed, QMoE per-channel, qk-norm fusion (= F + int8 lm_head) |
| `cuda_int4_int4_qmoe_kquant_mixed_lmh8_qknorm_qmoe64` | 11.61 | 24491.7 | 345.5 | 0.8170 | J: k_quant int4 body (blk32) + mixed int8 layers, int8 lm_head, FP16 embed, QMoE block64, qk-norm fusion (= E + int8 lm_head) |


## 8. Consolidated v2 sweep (A–J) — summary & recommendation

This section consolidates the qk-norm-fusion v2 builds (variants **A–J**) measured with the
**same harness** (Olive ModelBuilder built from source, `benchmark_e2e.py` batch 1 / prompt 512 /
gen 128 / CUDA graph=1 / **XQA=1**) and the **same evals commit (`db01e45`)** on the **full
14042-sample MMLU**. All ten are therefore directly comparable. (There is no variant "D" — the
letter was skipped.) Every variant shares: int4 body `block_size=32`, **FP16 embedding**,
qk-norm fusion on, `use_8bits_moe=0` (4-bit QMoE experts).

### 8.1 All variants, sorted by MMLU

| Var | Body algo | Mixed int8 layers | lm_head | QMoE blk | Size (GiB) | Prefill TPS | Decode TPS | MMLU |
|---|---|:--:|:--:|:--:|---:|---:|---:|---:|
| **J** | k_quant | yes | **int8** | 64 | 11.61 | 24491.7 | 345.5 | **0.8170** |
| **E** | k_quant | yes | int4 | 64 | 11.33 | 25664.2 | 363.7 | 0.8156 |
| **F** | k_quant | yes | int4 | per-ch (0) | 10.79 | 29018.5 | 366.7 | 0.8128 |
| **H** | k_quant | no  | int4 | 64 | 11.25 | 25037.2 | 369.0 | 0.8101 |
| **I** | k_quant | yes | **int8** | per-ch (0) | 11.06 | 28371.4 | 348.9 | 0.8093 |
| **G** | rtn | yes | int4 | 32 | 11.87 | 24421.5 | 374.7 | 0.8091 |
| **C** | rtn | yes | int4 | 64 | 11.31 | 25442.9 | 377.7 | 0.8087 |
| **A** | rtn | yes | int4 | per-ch (0) | 10.77 | 29403.1 | **382.0** | 0.8017 |
| **B** | rtn | no  | int4 | 64 | 11.23 | 25487.3 | 385.8 | 0.7901 |

> Decode leader overall is **B** (385.8) but at the lowest accuracy (0.7901); **A** is within ~1%
> (382.0) while scoring +0.0116 higher and being the smallest + best prefill — so A is the
> best *useful* speed point (see §8.3).

### 8.2 Isolated levers (each holds all other axes fixed)

| Lever | Comparison | Δ MMLU | Δ Decode | Δ Size | Δ Prefill |
|---|---|---:|---:|---:|---:|
| **k_quant vs rtn** (body) | F vs A / E vs C / H vs B | +0.011 / +0.007 / +0.020 | −15 / −14 / −17 | ~0 | ~0 |
| **mixed int8 layers** | C vs B / E vs H | +0.019 / +0.0055 | −8 / −5 | +0.08 GiB | ~0 |
| **QMoE 64 vs per-channel(0)** | C vs A | +0.007 | −4 | +0.54 GiB | **−13%** (29.4k→25.4k) |
| **QMoE 32 vs 64** | G vs C | −0.0004 (tie) | −3 | +0.56 GiB | −4% |
| **int8 vs int4 lm_head** | I vs F / J vs E | −0.0035 / +0.0014 | −18 / −18 | +0.27 GiB | −2% to −5% |

Takeaways:
- **k_quant** is the cheapest accuracy gain (+0.007–0.020 MMLU, no size cost), paid for in ~15 decode tps.
- **mixed int8 layers** reliably add accuracy for ~+0.08 GiB and a few decode tps — always worth it.
- **QMoE per-channel (0)** is the best size/prefill point; **block 64** buys ~+0.007 MMLU at −13% prefill
  and +0.54 GiB. **block 32 is strictly dominated** by 64 (no accuracy gain, bigger, slower).
- **int8 lm_head is not worth it**: accuracy change is within noise (−0.0035 to +0.0014) while costing
  ~+0.28 GiB and ~18 decode tps. Keep the int4 lm_head.

### 8.3 Recommendation — three-model lineup

A clean Pareto spread covering speed → balance → accuracy, all int4 lm_head except J:

| Role | Variant | Config | Size (GiB) | Prefill | Decode | MMLU |
|---|---|---|---:|---:|---:|---:|
| **Speed** | **A** `…rtn_mixed_lmh4_qknorm_qmoe0` | rtn, mixed, per-channel MoE, int4 lmh | **10.77** | **29403** | **382.0** | 0.8017 |
| **All-rounder** | **F** `…kquant_mixed_lmh4_qknorm_qmoe0` | k_quant, mixed, per-channel MoE, int4 lmh | 10.79 | 29018 | 366.7 | 0.8128 |
| **Accuracy** | **J** `…kquant_mixed_lmh8_qknorm_qmoe64` | k_quant, mixed, block-64 MoE, **int8 lmh** | 11.61 | 24492 | 345.5 | **0.8170** |

- **A (speed)** = the rtn twin of F; swapping k_quant→rtn (symmetric, no zero-points) is the decode lever.
  A leads decode **and** prefill **and** is the smallest, costing ~0.011 MMLU vs F.
- **F (all-rounder)** = best balance: near-top accuracy with the best size/prefill of the high-accuracy tier.
- **J (accuracy)** = the highest MMLU measured (0.8170), trading decode/prefill/size for the top score.
  (If the +0.28 GiB / ~18 decode of the int8 lm_head is unwanted, **E** at 0.8156 is the int4-lmh
  alternative — statistically indistinguishable accuracy, faster and smaller.)

---

## 9. Cross-build / cross-engine summary table (2026-06-26)

Requested side-by-side of the Foundry baseline (old released package vs. current source build,
CUDA graph on/off), the three recommended ORT variants (**A / F / J**), and llama.cpp.

- **Throughput** (Prefill = `pp512`, Decode = `tg128`): re-measured **fresh on 2026-06-26**,
  batch 1, prompt 512, gen 128, 5 reps / 2 warmup, GPU0 pinned. ORT rows via
  `benchmark_e2e.py`; llama.cpp via `llama-bench -ngl 99`.
- **MMLU**: **reused** from the full 14042-sample evals already recorded above (not re-run — it is
  expensive and runtime/library changes do not change accuracy for the same weights). Foundry =
  0.7929 (§3/§4), A/F/J = §8, llama.cpp = 0.8252 (§5.3, reasoning on, mt=768).
- **Builds**: source-build ORT rows use the current tree (onnxruntime `5f49a37`, genai
  `0.15.0.dev0`, provider lib built 2026-06-24, CUDA 13.0) with `ORT_ENABLE_XQA=1`. The baseline
  row uses the released wheel `onnxruntime-genai-cuda==0.13.1` with **onnxruntime-gpu 1.26.0**
  force-pinned (built for **CUDA 12.8**) in an isolated venv — no source lib synced. **ORT 1.27.0
  produces incorrect (gibberish) output on `foundry_cuda_v1`** (see §9.2), so it is *not* used as the
  baseline; ORT 1.26.0 is verified correct.

| Model | Size (GiB) | Prefill TPS | Decode TPS | MMLU | Comment |
|---|---:|---:|---:|---:|---|
| `foundry_cuda_v1` — **cg off, released pkg** (genai 0.13.1 / ORT 1.26.0, CUDA 12.8) | 11.04 | 29650.4 | **256.3** | 0.7929 | Baseline. `onnxruntime-genai-cuda==0.13.1` with **onnxruntime-gpu 1.26.0** force-pinned (the wheel's default dep is 1.27.0, which gives wrong output here — see §9.2). Output verified correct ("Paris") before benchmarking. No XQA / MoE-GEMV decode opts. CUDA-12.8 runtime via `LD_LIBRARY_PATH`. |
| `foundry_cuda_v1` — cg on, source build | 11.04 | 27987.1 | 366.8 | 0.7929 | Current source build (genai 0.15.0.dev0 / ORT 1.28, lib 06-24), XQA=1. +73% decode vs released baseline; +41% vs the old §4 number (260.0). |
| `foundry_cuda_v1` — cg off, source build | 11.04 | 30048.9 | 313.2 | 0.7929 | Same build, CUDA graph off. cg costs ~17% prefill but adds ~17% decode. |
| **A** `…rtn_mixed_lmh4_qknorm_qmoe0` | 10.77 | 29029.7 | 426.8 | 0.8017 | cg=1, XQA=1, source build. Decode leader. (§8 recorded 382.0 — see analysis below.) |
| **F** `…kquant_mixed_lmh4_qknorm_qmoe0` | 10.79 | 29656.9 | 409.2 | 0.8128 | cg=1, XQA=1. All-rounder. (§8 recorded 366.7.) |
| **J** `…kquant_mixed_lmh8_qknorm_qmoe64` | 11.61 | 24562.7 | 383.4 | 0.8170 | cg=1, XQA=1. Top MMLU. (§8 recorded 345.5.) |
| **llama.cpp** (MXFP4) | 11.28 | 11097.0 | 352.6 | 0.8252 | `llama-bench` build `039e20a2d` (9588), `-ngl 99`, pp512/tg128. Decode now 352.6 (was ~297 in §5.2 with older build). MMLU 0.8252 was with reasoning on, mt=768. |

### 9.1 Why the fresh ORT decode numbers are higher than the recorded values

The settings are **identical** to the recorded runs (batch 1, prompt 512, gen 128, CUDA graph on,
`ORT_ENABLE_XQA=1`). The difference is the **build**: the CUDA provider lib I measured was rebuilt
**2026-06-24**, which includes decode-path commits that landed **after** the §8 runs
(2026-06-21 → 06-22 00:30):

| Commit | Date | Effect |
|---|---|---|
| `6be94ded19` Enable XQA by default for FP16/BF16 GQA | 06-22 05:45 | XQA decode path on by default |
| `1472c16ea5` Enable CUDA GQA QK-Norm and XQA decode | 06-22 19:34 | fused qk-norm + XQA decode kernel |
| `669b8834fc` Sliding-window support to XQA decode | 06-23 21:21 | keeps XQA path active for GQA |
| `ba45260eed` Fuse MoE router bias into MatMulNBits GEMV | 06-24 02:07 | removes 24 router Add kernels (~+0.2%, see `qmoe_gemv_experiments.md`) |

Net effect: a uniform **~+11–12% decode** on A/F/J (382→427, 367→409, 345→383) with **prefill
essentially unchanged** (29403→29030, 29018→29657, 24492→24563) — confirming the gain is in the
**decode kernel** (XQA + GEMV), not prefill. The XQA-decode enablement (`1472c16ea5` /
`669b8834fc`) is the dominant driver; the router-bias GEMV fusion is small (~0.2%).

For Foundry the gap is larger (260.0 → 366.8, **+41%**) because its §4 number predates *all* of the
above — it was measured with the original `genai 0.14.0-dev0` study build (before qk-norm fusion and
XQA). The released-wheel baseline (`0.13.1` / **ORT 1.26.0**, verified-correct) sits lower at
**256.3** (cg off), giving the Foundry-weights ladder: **256.3 (0.13.1 / ORT 1.26.0, cg off) → 313.2
(06-24 source, cg off) → 366.8 (06-24 source, cg on)**.

> Note: the baseline used `onnxruntime-genai-cuda==0.13.1` with **onnxruntime-gpu 1.26.0**
> force-pinned (the wheel otherwise pulls 1.27.0). It is built for CUDA 12.8, so it was run with
> `LD_LIBRARY_PATH` pointed at `cuda12.8/lib64` + `cudnn9.12/lib` in a dedicated Python 3.12 venv.

### 9.2 ORT 1.27.0 correctness on `foundry_cuda_v1`

Before benchmarking the released wheels, a quick greedy-decode sanity prompt
("capital of France?") was run on each runtime against `foundry_cuda_v1`:

| Runtime | Output | Verdict |
|---|---|---|
| genai 0.13.1 + **ORT 1.26.0** | `" Paris\n\nSure! Here's a short…"` | ✅ correct |
| genai 0.13.2 + **ORT 1.27.0** | `" pulling us L tabletamar gainingser irault…"` | ❌ gibberish |

**ORT 1.27.0 produces incorrect output specifically on `foundry_cuda_v1`**, so its throughput
(measured earlier at decode 212.2) is meaningless and is excluded. ORT 1.27.0 is expected to work
correctly on other models — the released 0.13.x wheels could not load the **A/F/J** variants for an
independent check (their tokenizer uses a newer `TokenizersBackend` class unsupported by the 0.13.x
runtime, `RuntimeError: Unsupported tokenizer class`), and the A/F/J throughput rows above come from
the current source build (ORT 1.28-dev), not 1.27.0. The baseline therefore uses the
verified-correct **ORT 1.26.0**.

---

## 11. GPT-OSS-20B QMoE per-channel storage fix and 16-bucket experiment (2026-07-02 to 2026-07-03)

This section records the rc1 rebuild/debug session for
`gpt-oss-20b/cuda/gpt-oss-20b_cuda_int4_int4_qmoe_rtn_mixed_matmul_only_qknorm_bs0.json`.
The key issue was wrong output from the newly rebuilt model compared with the archived known-good model:

- Archived known-good model:
  `gpt-oss-20b/cuda/variants/cuda_int4_int4_qmoe_rtn_mixed_matmul_only_qknorm_bs0`
- Rebuilt/fixed models:
  - `gpt-oss-20b/cuda/model_qmoe_unsigned_offset` — unsigned-offset, legacy 15-bucket range
  - `gpt-oss-20b/cuda/model_qmoe_trtllm_signed` — TRT-LLM-style signed/two's-complement storage experiment
  - `gpt-oss-20b/cuda/model_qmoe_unsigned_full_range` — unsigned-offset, full 16-bucket range
- Repro script added under the dev repo:
  `/home/tianlei/dev/scripts/h200_18/run_gpt_oss_rc1.sh`

### 11.1 Reproduction commands

Use the source-build ORT/genai CUDA 13.0 environment:

```bash
export CUDA_HOME=/home/tianlei/cuda13.0
export CUDA_PATH=/home/tianlei/cuda13.0
export CUDNN_HOME=/home/tianlei/cudnn_9.19_cuda13
export ORT_BUILD_DIR=/home/tianlei/onnxruntime/build/cu130_bench/Release
export VENV=/home/tianlei/onnxruntime/.venv_cu130
export ORT_ENABLE_XQA=1
export CUDA_VISIBLE_DEVICES_BENCH=0
export MMLU_GPUS="0..7"
export MMLU_PARALLEL=1
export MMLU_MAX_SAMPLES=800

# Rebuild genai, rebuild the Olive model, run the capital-of-France sanity check,
# benchmark, and run MMLU. By default this now emits 16-bucket unsigned-offset QMoE.
/home/tianlei/dev/scripts/h200_18/run_gpt_oss_rc1.sh --all

```

For MMLU stability during this session, export `ORT_FORCE_DETERMINISTIC_MOE=1`. Without it, the
non-deterministic/fused MoE path can hit a CUDA illegal-memory-access failure during eval (see
§11.5).

### 11.2 Root cause and code changes

Two issues were found in the rebuilt QMoE model path:

1. **Initializer shape bookkeeping:** GPT-OSS QMoE initializer shape metadata must follow the actual
   stacked qweight tensor shapes, especially for per-channel QMoE (`qmoe_block_size <= 0`) and
   CUDA-prepacked weights.
2. **QMoE storage contract:** ORT CUDA QMoE prepack/runtime decodes raw int4 as `nibble - 8` and raw
   int8 as `byte - 128`. Therefore the exported QMoE storage must be unsigned offset (`q + 8` /
   `q + 128`) even though numeric quantization is symmetric. A TRT-LLM-style signed/two's-complement
   byte/nibble encoding is not compatible with this runtime path.

The helper quantizer was added/updated in onnxruntime-genai and mirrored in ORT test tooling:

- GenAI helper: `/home/tianlei/onnxruntime-genai/src/python/py/models/builders/qmoe_quantizer.py`
- GenAI builder call sites: `/home/tianlei/onnxruntime-genai/src/python/py/models/builders/base.py`
- GPT-OSS shape fix: `/home/tianlei/onnxruntime-genai/src/python/py/models/builders/gptoss.py`
- GenAI focused tests: `/home/tianlei/onnxruntime-genai/test/python/builder/test_qmoe_weights.py`
- ORT helper: `/home/tianlei/onnxruntime/onnxruntime/python/tools/quantization/qmoe_quantizer.py`
- ORT CUDA smoke: `/home/tianlei/onnxruntime/onnxruntime/test/python/transformers/test_qmoe_cuda.py`

Final supported QMoE per-channel modes:

| Mode | Numeric range | Scale | Stored value | Status |
|---|---|---|---|---|
| unsigned full range | int4 `[-8, 7]`, int8 `[-128, 127]` | `/8`, `/128` | `q + 8`, `q + 128` | default |
| unsigned legacy range | int4 `[-7, 7]`, int8 `[-127, 127]` | `/7`, `/127` | `q + 8`, `q + 128` | env-var testing only |
| TRT-LLM signed storage | int4 `[-8, 7]`, int8 `[-128, 127]` | `/8`, `/128` | two's-complement signed bits | removed; broken |

### 11.3 Validation

Focused validation run after the cleanup:

```bash
cd /home/tianlei/onnxruntime-genai
/home/tianlei/onnxruntime/.venv_cu130/bin/python -m pytest test/python/builder/test_qmoe_weights.py -q
# 32 passed, 2 warnings

cd /tmp
export CUDA_HOME=/home/tianlei/cuda13.0 CUDA_PATH=/home/tianlei/cuda13.0
export CUDNN_HOME=/home/tianlei/cudnn_9.19_cuda13 ORT_ENABLE_XQA=1 ORT_FORCE_DETERMINISTIC_MOE=1
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/tianlei/onnxruntime/build/cu130_bench/Release:/home/tianlei/onnxruntime/onnxruntime/test/python/transformers
export LD_LIBRARY_PATH=/home/tianlei/onnxruntime/build/cu130_bench/Release:/home/tianlei/onnxruntime/build/cu130_bench/Release/onnxruntime/capi:/home/tianlei/cuda13.0/lib64:/home/tianlei/cudnn_9.19_cuda13/lib64:/home/tianlei/cudnn_9.19_cuda13/lib:${LD_LIBRARY_PATH:-}
/home/tianlei/onnxruntime/.venv_cu130/bin/python -m pytest /home/tianlei/onnxruntime/onnxruntime/test/python/transformers/test_qmoe_cuda.py::TestQMoEIntPrePackSmoke -q
# 5 passed, 2 subtests passed

bash -n /home/tianlei/dev/scripts/h200_18/run_gpt_oss_rc1.sh
```

### 11.4 MMLU storage-mode experiment

All MMLU rows here are capped at 800 samples (`MMLU_MAX_SAMPLES=800`). Artifacts are under
`/tmp/gpt_oss_qmoe_storage_mmlu_800`.

| Model / run | QMoE storage | Env | MMLU result | Artifact |
|---|---|---|---:|---|
| `model_qmoe_unsigned_offset` | unsigned offset, 15-bucket | `ORT_FORCE_DETERMINISTIC_MOE=1` | **0.8488** (679/800) | `/tmp/gpt_oss_qmoe_storage_mmlu_800/unsigned_offset_deterministic_moe/mmlu.summary` |
| `model_qmoe_unsigned_full_range` | unsigned offset, 16-bucket | `ORT_FORCE_DETERMINISTIC_MOE=1`, 8 GPUs | **0.8525** (682/800) | `/tmp/gpt_oss_qmoe_storage_mmlu_800/unsigned_full_range_mmlu_8gpu/mmlu.summary` |
| `model_qmoe_trtllm_signed` | TRT-LLM signed/two's-complement | `ORT_FORCE_DETERMINISTIC_MOE=1` | failed / invalid | `/tmp/gpt_oss_qmoe_storage_mmlu_800/trtllm_signed_deterministic_moe/mmlu.log` |

Conclusion: the 16-bucket unsigned-offset mode slightly improves the 800-sample score over the
legacy 15-bucket unsigned-offset mode and keeps the ORT CUDA QMoE storage contract. The signed
storage mode is empirically broken and was removed from the code and docs.

### 11.5 QMoE profiler cross-stream race (RESOLVED 2026-07-03)

**Symptom.** The MMLU runner (and a minimal genai repro) crashed intermittently unless
`ORT_FORCE_DETERMINISTIC_MOE=1` was set. Disabling CUDA graph alone also avoided it. The visible
failure usually surfaced later as a downstream launch failure + CUDA 700 (illegal memory access),
e.g. `/lm_head/MatMul_Q8` cuBLAS launch failure, or inside the MoE grouped GEMM itself:

```text
moe_gemm_template_dispatch.h:194 ... occupancy > 0 was false. GPU lacks the shared memory resources
moe_kernels.cu:344 ... cudaFuncSetAttribute(...) CUDA failure 700: an illegal memory access
```

The crashing layer varied run-to-run (layers 0/1/7…) — a classic non-deterministic race signature.

**Isolation.** Reproduced deterministically (≈100% crash) with an 8×H200 (sm_90) build, CUDA 13.0,
`ORT_ENABLE_XQA=1`, `enable_cuda_graph=1`, model `model_qmoe_unsigned_full_range` (INT4 per-channel,
`block_size=-1`). Key facts:

- `ORT_FORCE_DETERMINISTIC_MOE=1` (profiler never runs) → always clean.
- `enable_cuda_graph=0` → always clean.
- `CUDA_LAUNCH_BLOCKING=1` and `compute-sanitizer` (both serialize launches) → always clean, 0
  memcheck errors → confirms a pure concurrency/timing bug, not a static OOB.
- Draining the compute stream *before* profiling did **not** help (the hazard is with *subsequent*
  reuse of the scratch block, not prior writes).

**Root cause.** The MoE GEMM profiler (`MoeGemmProfiler::runProfiling`) allocated scratch from the
per-node **temp allocator** but ran its grouped-GEMM/routing kernels on a **private side stream**.
The temp arena is stream-aware and tracks liveness on the compute stream; the profiler's side-stream
usage is invisible to it, so the arena could hand the same scratch block to a later compute-stream
allocation (the real MoE workspace) while the profiler's kernels were still in flight. The
overlapping access corrupted the profiler's routing/GEMM buffers and left a sticky CUDA 700 that
surfaced at the next MoE (or `lm_head`) kernel launch. This only manifested with CUDA graph enabled
(which changes the warmup/allocation ordering) and on SM90 (the INT4 mixed-input non-TMA tactics).

A second, independent latent bug was found and fixed in `runProfiler`: its workspace layout was keyed
on the per-tactic `tactic.is_tma_warp_specialized`, whereas `getWorkspaceSize` and every
`prepare*` helper key it on `mSM >= 90`. On SM90 with a non-TMA (Ampere-fallback) INT4 tactic these
diverge, shifting all sub-buffer offsets so the profiler reads `expert_first_token_offset` and the
GEMM inputs from wrong, random-filled locations.

**Fix (onnxruntime `tlwu/20260701/qmoe_fp4`).**

1. `moe_gemm_profiler.{h,cc}`: `profileTactics`/`runProfiling` take an optional `timing_stream`; when
   provided the profiler runs on it (the ORT compute stream) so all profiler kernels are strictly
   ordered with the surrounding compute-stream work and share its temp-allocator stream context. A
   private stream is created only when no stream is supplied.
2. `moe_quantization.cc`: pass the compute stream (`Stream(context)`) into `profileTactics`; skip
   profiling entirely while the compute stream is being captured into a CUDA graph
   (`isCapturing`), with a capture-safe fallback to `tactics[0]` when no tuned config is cached.
3. `moe_kernels.cu`: key `runProfiler`'s `getProfilerWorkspaces` layout on `mSM >= 90` to match
   `getWorkspaceSize`/`prepare*`.

**Verification.** Minimal genai repro (`/tmp/qmoe_repro/gen.py`, long prefill, 3 iters, GPU0),
CUDA-graph on: **15/15** clean runs after the fix (was ≈100% crash before). Controls still pass:
`ORT_FORCE_DETERMINISTIC_MOE=1` 3/3, `enable_cuda_graph=0` 3/3. `lintrunner` clean on all four files.


---

## 10. MoE GEMV FP32 accumulation: `ORT_MOE_GEMV_FP32_ACCUM` (2026-06-27)

Effect of accumulating the int4 QMoE GEMV (decode-path MoE expert mat-vec) in **FP32** vs **FP16**,
toggled by the env var `ORT_MOE_GEMV_FP32_ACCUM` (`0` = FP16 accum, `1` = FP32 accum), on the two
recommended variants **F** (`…kquant_mixed_lmh4_qknorm_qmoe0`) and **J** (`…kquant_mixed_lmh8_qknorm_qmoe64`).

- **Binary**: fresh source build at `onnxruntime/build/cu130_bench/Release` (onnxruntime `c3a5222d2a`,
  ORT 1.28-dev, CUDA 13.0), provider lib synced into the `.venv_cu130` genai package.
- **Throughput**: `benchmark_e2e.py`, batch 1, prompt 512, gen 128, 5 reps / 2 warmup, GPU0 pinned,
  `enable_cuda_graph=1`, `ORT_ENABLE_XQA=1`.
- **MMLU**: full **14,042**-sample `match_mmlu`, 8-GPU sharded (`run_evals_parallel.sh`), reasoning
  medium, `max_new_tokens=2048`, with the env var exported to all workers.
- **Correctness**: a greedy sanity prompt ("capital of France?") returned " Paris" for all four
  setups before benchmarking.

| Model | FP32_ACCUM | Size (GiB) | Prefill TPS | Decode TPS | MMLU | Comment |
|---|:---:|---:|---:|---:|---:|---|
| **F** `…kquant_mixed_lmh4_qknorm_qmoe0` | 0 (FP16) | 10.79 | 29157.8 | **410.6** | 0.8051 | Decode leader; FP16 MoE-GEMV accum. |
| **F** `…kquant_mixed_lmh4_qknorm_qmoe0` | 1 (FP32) | 10.79 | 29015.2 | 374.7 | 0.8059 | +0.0008 MMLU for −8.8% decode. |
| **J** `…kquant_mixed_lmh8_qknorm_qmoe64` | 0 (FP16) | 11.61 | 25136.5 | **385.9** | 0.8185 | FP16 MoE-GEMV accum. |
| **J** `…kquant_mixed_lmh8_qknorm_qmoe64` | 1 (FP32) | 11.61 | 24315.1 | 353.0 | 0.8200 | +0.0015 MMLU for −8.5% decode. |

**Takeaways**

1. **FP32 accumulation barely changes accuracy**: F +0.0008 (0.8051 → 0.8059), J +0.0015
   (0.8185 → 0.8200) on the full 14,042-sample MMLU — within run-to-run noise.
2. **FP32 accumulation costs ~9% decode**: F 410.6 → 374.7 (−8.8%), J 385.9 → 353.0 (−8.5%);
   prefill is essentially unchanged (the GEMV is a decode/mat-vec kernel, not used in prefill GEMMs).
3. **Recommendation**: keep the default **FP16 accumulation** (`ORT_MOE_GEMV_FP32_ACCUM=0`) — the
   ~9% decode speedup is free given the negligible accuracy delta. Reserve FP32 accum for cases that
   demand maximum numerical fidelity.

---

## 12. v2 prepack + block-size sweep on ORT 1.29 / genai 0.15 (2026-07-09)

New model-builder line ("v2") that drives prepacking through recipe `extra_options`
(`matmulnbits_weights_prepacked`, `qmoe_weights_prepacked`, `qmoe_block_size`) instead of env vars,
plus the `fuse_qk_norm_gqa` fusion. This section sweeps QMoE block size, the dense MatMulNBits
prepack mode (SM80 vs SM90 fpA_intB layout), the LM-head bit width, and the int4 algo, and measures
full-MMLU accuracy + throughput.

### Stack / method
- **ORT**: 1.29.0, source build `~/git/onnxruntime/build/cu130_bench/Release`
  (wheel `onnxruntime_gpu-1.29.0`), CUDA 13.0, 8×H200 (sm_90).
- **onnxruntime-genai**: 0.15.0-dev built from source, branch `tlwu/update_model_builder_gpt_oss`
  (HEAD `7fc783726c`). Clean-reinstall the wheel after every build (a stale
  `site-packages/onnxruntime_genai/models/builders/` is NOT overwritten by `pip --force-reinstall`;
  `pip uninstall` + `rm -rf` the package dir first, then reinstall, and verify from a neutral cwd).
- **venv**: `~/git/onnxruntime/.venv_cu130` (python 3.14).
- **Recipes**: `gpt-oss-20b_v2_rc{0..10}_*.json` in `gpt-oss-20b/cuda/`. Models in
  `~/gpt_oss_rc_models/v2_rc{0..10}/`. Runner `~/git/dev/scripts/h200_18/run_gpt_oss_rc1.sh`.
- **Benchmark**: `benchmark_e2e.py`, batch 1, prompt 512, gen 128 (reps 5–10 / warmup 2–3), GPU0,
  `enable_cuda_graph=1`, `ORT_ENABLE_XQA=1`. **Model size** = `model.onnx` + `model.onnx.data`.
- **MMLU**: full **14,042**-sample `match_mmlu`, multi-GPU sharded (`run_evals_parallel.sh`).
  RC9/RC10 sharded on 6 GPUs (another user occupied GPU 1/2); all others on 8 GPUs.
- **Common v2 extra_options**: `fuse_qk_norm_gqa=1`, `qmoe_weights_prepacked=1`,
  `int8_mixed_layers=1`, `use_8bits_moe=0`, `int4_algo_config=default` (except rc5/rc7 `k_quant`).

### How the dense-weight prepack works (current ORT)
`prepack_matmulnbits_weights` (genai `base.py`) repacks eligible **symmetric** MatMulNBits qweights
into the CUDA fpA_intB layout and stamps `weight_prepacked` on the node:
- `matmulnbits_weights_prepacked = 0` → raw weights (standard MatMulNBits path).
- `= 1` → **SM80** layout (`weight_prepacked=1`), eligible `block_size ∈ {32,64,128}`.
- `= 2` → **SM90** (Hopper native) layout (`weight_prepacked=2`), eligible `block_size ∈ {64,128}` only.
Eligibility also needs `bits∈{4,8}`, `K%block_size==0`, `N%(32 int8 / 64 int4)==0`, symmetric weights.
On sm_90, `FpAIntBPackingSmForKernel()` returns 90 only for `weight_prepacked==2`, else 80; the
kernel `ORT_ENFORCE`s the model's format matches. gpt-oss has 73 MatMulNBits nodes; the 24 MoE
routers (N=32) are never eligible, so at most **49/73** prepack (24 o_proj + 24 qkv + 1 lm_head).
fpA_intB now auto-engages (no `ORT_FPA_INTB_GEMM` needed) and supports fused bias.

### 12.1 QMoE block size, LM-head bits, prepack mode, algo (rc0–rc5, dense `int4_block_size=32`)

| rc  | recipe suffix              | QMoE blk | lm_head | prepack | algo    | Size    | Prefill | Decode | MMLU |
|-----|----------------------------|:--------:|:-------:|:-------:|---------|--------:|--------:|-------:|-----:|
| rc0 | `default_prepack0`         | 0 (per-ch) | int8 | 0      | default | 11.8 GB | 18675 | 411.9 | 0.8074 |
| rc1 | `default_prepack1`         | 0        | int8    | 1 (SM80)| default | 11.8 GB | 18158 | 421.4 | 0.8056 |
| rc2 | `default_prepack2`         | 0        | int8    | 2 (SM90)| default | 11.8 GB | 18550 | 412.5 | 0.8074 |
| rc3 | `default_prepack0_lm4`     | 0        | int4    | 0      | default | 11.5 GB | 18624 | 436.6 | 0.8054 |
| rc4 | `default_prepack1_bs64`    | **64**   | int8    | 1 (SM80)| default | 12.4 GB | 15047 | 420.8 | **0.8211** |
| rc5 | `kquant_prepack0`          | 0        | int8    | 0      | k_quant | 11.9 GB | 18798 | 389.9 | 0.8087 |

- **QMoE block-wise (bs64) is the single biggest accuracy lever**: rc4 0.8211 vs the per-channel
  (bs0) group ~0.805–0.808 (+~1.3 pts). MoE experts dominate the parameter count, so finer MoE
  scales matter most. Cost: +0.6 GB and −19% prefill (15047 vs ~18.5k; block-wise dequant is heavier),
  decode ~unchanged.
- **Prepack mode is accuracy-neutral** (layout only): rc0=rc2=0.8074 (mode 0 vs 2 on bs32/per-ch MoE
  are the same weights); rc1 0.8056 is noise. It only shifts prefill/decode.
- **rc3 `lm_head int4`** (`last_matmul_weight_int8=0`): fastest decode (436.6) + smallest (11.5 GB)
  but lowest MMLU (0.8054) — the int8 LM head is worth ~1 pt.
- **rc5 k_quant**: 0.8087, marginally above default per-channel; slowest decode (389.9). k_quant is
  **asymmetric** (per-block zero-points) so it can NOT be prepacked (fpA_intB is symmetric-only).

### 12.2 LM-head off + k_quant×bs64 (rc6, rc7; dense `int4_block_size=32`)

| rc  | recipe suffix                | vs rc4              | Size    | Prefill | Decode | MMLU |
|-----|------------------------------|---------------------|--------:|--------:|-------:|-----:|
| rc6 | `default_prepack1_bs64_lm4`  | rc4 + lm_head int4  | 12.1 GB | 15141 | 431.2 | 0.8190 |
| rc7 | `kquant_prepack0_bs64`       | k_quant + QMoE bs64 | 12.4 GB | 15160 | 387.3 | 0.8210 |

- **rc6**: −0.3 GB, +2.5% decode (420.8→431.2) for −0.2 pt MMLU (0.8190). Balanced pick.
- **rc7**: MMLU 0.8210 ≈ rc4 0.8211 → **k_quant body does NOT stack on top of QMoE bs64** (the MoE
  block-wise gain already captured the accuracy; dense algo is a wash). Worst decode (387, asymmetric)
  and unprepackable → **dominated by rc4**; drop k_quant for this line.

### 12.3 Dense block size 32 vs 64 × SM80 vs SM90 prepack (rc8, rc9, rc10)

All keep QMoE bs64 and lm_head int8 (= rc4), varying only the dense `int4_block_size` and prepack mode.

| rc   | dense int4_blk | prepack | Prepacked | Size    | Prefill | Decode | MMLU |
|------|:--------------:|:-------:|:---------:|--------:|--------:|-------:|-----:|
| rc4  | 32             | 1 (SM80)| 49×wp1    | 12.4 GB | 15047 | 420.8 | **0.8211** |
| rc8  | 32             | 2 (SM90)| **0**×wp2 | 12.4 GB | 15350 | 407.3 | (not run) |
| rc9  | 64             | 1 (SM80)| 49×wp1    | 12.4 GB | 15083 | 419.4 | 0.8093 |
| rc10 | 64             | 2 (SM90)| 49×wp2    | 12.4 GB | 15902 | **460.95** | 0.8089 |

- **rc8 is a no-op prepack trap**: SM90 mode needs block_size 64/128, but the dense weights are
  bs32 → **0/73 prepacked**, silently falling back to the standard path (decode 407 < rc4's fpA_intB
  SM80 GEMV 421 in a clean same-GPU A/B). SM90 mode requires `int4_block_size=64` to engage.
- **rc10 reproduces the SM90 native decode win**: with bs64 the 49 nodes pack as SM90 (`weight_prepacked=2`)
  → decode **460.95 vs rc9 (SM80) 419.4 = +9.9%**, prefill +5.4%. Matches the earlier v1 SM90 result.
- **Prepack layout is lossless**: rc9 and rc10 are byte-identical in size and MMLU (0.8093 vs 0.8089,
  noise); SM80 vs SM90 is purely perf.
- **bs64 costs ~1.2 MMLU pts vs bs32** on the full eval: rc9/rc10 ~0.809 vs rc4 (bs32) 0.8211. Finer
  bs32 scales quantize the attention/lm_head weights better. (Earlier MMLU-800 "bs32≈bs64 wash" was
  small-sample noise.) Because SM90 prepack *requires* bs64, its +10% decode is inseparable from this
  ~1.2-pt accuracy loss.

### Conclusions / recommendations
1. **Best accuracy: rc4** (`default`, QMoE bs64, dense bs32, SM80 prepack, int8 lm_head) — **0.8211**,
   decode 420.8. QMoE block-wise + int8 lm_head + fine dense bs32 is the accuracy recipe.
2. **Lowest latency: rc10** (dense bs64, SM90 prepack) — decode **461** (+9.6% vs rc4) but **−1.2 pt
   MMLU** (0.8089); the SM90 native kernel is only reachable at bs64.
3. **Balanced: rc6** (rc4 + int4 lm_head) — 0.8190, decode 431, 12.1 GB.
4. **Drop**: rc7 (k_quant no gain, unprepackable, slow), rc8 (SM90 mode with bs32 = 0 prepacked),
   rc9 (dominated: rc4's speed at rc4-minus-1.2pt accuracy). k_quant can't be prepacked (asymmetric).
5. To get SM90 decode at bs32 accuracy would require an **ORT-side** change letting the SM90 fpA_intB
   prepack/kernel accept `block_size=32` — not a recipe knob.

### 12.4 SM90 native `block_size=32` prepack enabled — rc8 rebuilt (2026-07-10)

This delivers the **ORT-side change from conclusion #5**: the native SM90 (Hopper TMA/WGMMA)
fpA_intB mixed-GEMM kernel now serves `block_size=32` via a multi-scale-per-K-tile
(`ScaleKPerTile=2`) mainloop (two bs32 scale groups share the 64-element Hopper K-tile). So the
"SM90 requires bs64" trap from 12.3 is gone: the rc8 recipe (`v2_rc8_default_prepack2_bs64.json`,
dense `int4_block_size=32`, `matmulnbits_weights_prepacked=2`) now actually prepacks.

**What changed**
- **ORT** (branch `tlwu/20260710/fpa_intb_sm90_bs32_inmem_autotune`): SM90 mixed-GEMM mainloop +
  launcher gained the `ScaleKPerTile=2` path; the `MatMulNBits` ctor now allows
  `weight_prepacked=2` with `block_size ∈ {32,64,128}`. 64/128 kernels are byte-identical
  (`if constexpr (ScaleKPerTile==1)`-gated), so no perf/accuracy impact on rc9/rc10.
- **genai** `base.py` `prepack_matmulnbits_weights`: SM90 (`prepack_mode==2`) `allowed_block_sizes`
  widened from `{64,128}` → `{32,64,128}` (weight byte-layout is block-size-independent; bs32 is a
  runtime scale-grouping concern). This **supersedes** the "SM90 eligible `{64,128}` only" line in
  *How the dense-weight prepack works* above.

**Rebuilt model** — `~/gpt_oss_rc_models/v2_rc8_prepack2/` (12.4 GB), same 8×H200 / ORT 1.29 /
CUDA 13.0 stack (benchmark: batch 1, prompt 512, gen 128, reps 10 / warmup 3, GPU0,
`enable_cuda_graph=1`, `ORT_ENABLE_XQA=1`):

| rc          | dense int4_blk | prepack  | Prepacked  | Size    | Prefill | Decode | MMLU |
|-------------|:--------------:|:--------:|:----------:|--------:|--------:|-------:|-----|
| rc4 (ref)   | 32             | 1 (SM80) | 49×wp1     | 12.4 GB | 15047 | 420.8 | 0.8211 *(full)* |
| rc10 (ref)  | 64             | 2 (SM90) | 49×wp2     | 12.4 GB | 15902 | 460.95 | 0.8089 *(full)* |
| rc8 (old)   | 32             | 2 (SM90) | **0**×wp2  | 12.4 GB | 15350 | 407.3 | (fell back, 12.3) |
| **rc8 (new)** | 32           | 2 (SM90) | **49**×wp2 | 12.4 GB | **15931** | **457.6** | **0.8600** *(800)* |

- **49/73 MatMulNBits now pack as native SM90 bs32** (`weight_prepacked=2`): 24 qkv (N=2880) +
  24 o_proj/mlp (N=5120, int4) + 12 int8-mixed (N=5120) + 1 lm_head (N=201088, int8). The 24 MoE
  routers (N=32, `N%64≠0`) remain ineligible — exactly the expected `49/73`.
- **Generation correct** (m=1 GEMV decode on the SM90 layout): "The capital of France is **Paris**."
- **MMLU 0.8600 (688/800)** — a **800-sample** run, so *not* directly comparable to the full-eval
  (14,042) numbers in 12.1–12.3. Prepack is a lossless layout change (established by rc9≡rc10), so
  this confirms the SM90 bs32 kernel is numerically clean vs the raw bs32 weights (no
  regression/corruption); it anchors against the earlier rc7_v2 800-sample baseline (0.8612).
- **The A/B the SM90-bs32 kernel was built for** (all bs32, same weights, layout-only diff): decode
  **457.6 vs rc4's SM80 bs32 420.8 = +8.8%**, prefill **15931 vs 15047 = +5.9%** — and it lands right
  on rc10's SM90 numbers (decode 460.95, prefill 15902, ≈noise) **while keeping bs32 accuracy**. So
  the native SM90 kernel now delivers rc10-class throughput *without* rc10's bs64 −1.2-pt MMLU hit
  — exactly the goal of conclusion #5.

**Upshot for conclusions #2/#5**: SM90 native decode is no longer locked to bs64. rc8 (new) combines
rc4's bs32 accuracy with rc10's SM90 decode speedup (decode 457.6 ≈ rc10's 461, +8.8% over rc4;
accuracy stays at the bs32 tier, not rc10's −1.2-pt bs64 drop) — making it the new **best
accuracy+latency** candidate. Confirm with a full-MMLU (14,042) pass before promoting over rc4.

## 13. Quantized KV cache — INT8 / FP8 / INT4, calibration + full range + leaderboard evals (2026-07-12)

Adds **KV-cache quantization** to the gpt-oss-20b CUDA `GroupQueryAttention` node (post-RoPE K and
raw V), on top of the existing int4-weight `default_prepack` recipe. All four models below share
**byte-identical int4 weights** (recipe `gpt-oss-20b_rc7_default_prepack.json`, `qmoe_block_size=0`
per-channel, prepacked; `model.onnx.data` = 11,821,236,224 B for every one) and differ **only in the
KV-cache element type** — an apples-to-apples isolation of KV quantization. Stack: 8×H200, ORT 1.29
(build `cu130_bench`, `USE_FPA_INTB_GEMM/USE_FP8_KV_CACHE/USE_INT4_KV_CACHE=ON`), genai `tlwu/quantized_kv_cache`,
CUDA 13.0 / cuDNN 9.19.

### 13.1 Calibration & scale convention

- Scales calibrated from the FP16-KV baseline's `present.*.key/value` over 24 diverse **512-token**
  sequences (percentile 99.99, per-channel). Script: `dev/scripts/h200_18/calibrate_kv_scales.py`;
  orchestrator `run_gpt_oss_quantized_kv.sh`.
- **Root cause of the earlier INT8 gap:** old scales were calibrated on ~40-token prompts, but eval
  runs at 512-token context. With RoPE, post-RoPE K amax grows with position, so short-prompt calib
  under-estimated amax on ~90 % of channels → the deployed model hard-clipped KV at long context.
  512-token calibration closes it (INT8 MMLU-800 0.7512 → 0.8575).
- **Use the full signed range (`unsigned_full_range`): `scale = amax / 2^(b-1)`, clamp
  `[-2^(b-1), 2^(b-1)-1]`.** The ORT GQA kernel (`group_query_attention_qdq.cuh`) clamps INT4 to
  `[-8,7]` (`kInt4Min/Max`) and INT8 to `[-128,127]` (`kInt8Min/Max`) and dequantizes `q·scale`, so
  the scale must divide by `2^(b-1)` (8 / 128), not `2^(b-1)-1` (7 / 127). This uses all `2^b` levels.
  Same convention as the weight quantizer's `unsigned_full_range=True` default (`cuda_quantizer.py`).
  - **INT4 `/8` vs `/7`: +5.1 pt** (MMLU-800 0.7788 → 0.8300) — the ~12.5 % finer step dominates.
  - **INT8 `/128` vs `/127`: neutral** (0.8575 → 0.8538, within ±1σ) — 0.4 % finer step is noise.
  - fp8 is unaffected (symmetric ±448, floating grid).
- Models tagged `*_percentile_fr` use the full-range convention. Scale JSONs in
  `~/gpt_oss_kv_scales/kv_scales_<q>_per_channel_percentile_fr.json`.

### 13.2 Models

| tag | KV cache | model dir (`~/gpt_oss_rc_models/`) | cache bytes/elem |
|-----|----------|------------------------------------|:----------------:|
| base    | FP16 (baseline)         | `rc7_v2_fp16kv_default`         | 2   |
| int8-fr | INT8 per-ch, `[-128,127]` | `rc7_v2_int8_kv_percentile_fr` | 1   |
| fp8     | FP8 E4M3 per-ch         | `rc7_v2_fp8_kv_percentile`      | 1   |
| int4-fr | INT4 per-ch, `[-8,7]`   | `rc7_v2_int4_kv_percentile_fr`  | 0.5 |

`base` was **freshly rebuilt** from `gpt-oss-20b_rc7_default_prepack.json` with the current model
builder (not the older `rc7`/`rc7_v2` k-quant/bs64 models, which use different weight quant and are
**not** comparable). On-disk size is 11.04 GiB for all four — KV quant only changes the runtime cache
dtype, not stored weights (the memory saving is in the GPU KV cache during inference).

### 13.3 Results — throughput + leaderboard accuracy

Benchmark: `benchmark_e2e.py`, batch 1 / prompt 512 / gen 128, reps 10 / warmup 3, GPU0,
`enable_cuda_graph=1`, `ORT_ENABLE_XQA=1`. Evals via `run_gpt_oss_rc1_v2.sh` (OpenAI-evals stack,
`oss/gpt-oss-20b*` completion fns, 8-GPU sharding).

| tag | Prefill tps | Decode tps | GPQA-diamond (198) | MATH (500) | **MMLU-Pro (12,032)** |
|-----|--------:|-------:|-----:|-----:|-----:|
| **base** FP16 | 18,157 | 424.3 | 0.5455 | 0.8740 | **0.6841** (8231) |
| **int8-fr**   | 18,998 | 351.8 | 0.5859 | 0.8620 | **0.6828** (8215) |
| **fp8**       | 19,007 | 349.7 | 0.5404 | 0.8680 | **0.5636 ⚠︎** (6781) |
| **int4-fr**   | 18,675 | 351.5 | 0.5000 | 0.8040 | **0.6271** (7545) |

Supporting MMLU-800 (per-channel, 512-tok percentile, full-range S0): base 0.8612 · int8-fr 0.8538 ·
fp8 0.8588 · int4-fr 0.8300.

### 13.4 Findings & recommendation

- **INT8-fr ≈ FP16 and is the recommended candidate.** MMLU-Pro 0.6828 vs 0.6841 (−0.13 pt, noise);
  GPQA/MATH/MMLU-800 all within noise. Half the KV-cache memory, fully stable (no runtime issues),
  decode ~352 tps.
- **FP8 is near-lossless *with cuda graph on*** (GPQA 0.5404, MATH 0.8680, MMLU-800 0.8588 all ≈ base)
  **but the FP8 KV path is not production-ready** — see the ⚠︎ MMLU-Pro number and §13.5.
- **INT4-fr has a real reasoning hit**: MMLU-Pro −5.7 pt (0.6271), MATH −7.0 pt (0.8040), GPQA −4.6 pt
  — larger than the ~3-pt drop on plain MMLU. The coarse 4-bit grid costs more on hard reasoning, even
  at a quarter of the KV memory. Full-range `[-8,7]` is mandatory (without it, 0.7788).
- **Throughput/size**: all quantized variants have ~equal or slightly higher prefill than FP16 and
  ~350 tps decode (vs FP16 424 — dequant overhead); on-disk size is identical.

### 13.5 Work items / hand-off

- **[OPEN] FP8 KV cache is cuda-graph-dependent and unstable — blocks FP8 promotion.**
  Two distinct symptoms on `rc7_v2_fp8_kv_percentile` during long-generation MMLU-Pro:
  1. **`enable_cuda_graph=1` → intermittent `Bus error` (SIGBUS, core dumped, exit 135/139)** in the
     genai runtime mid-eval (all 8 shards hit it near the same point). *Not* resource exhaustion
     (`/tmp` 8 %, shm 1 %, 1.7 TB RAM free). The par-eval harness's checkpoint/resume then enters a
     **non-converging retry loop** (two partial resume files `resume_from_250` / `resume_from_1254`
     never merge → shard relaunches indefinitely). base/int8/int4 completed the full 12,032 cleanly
     with cuda graph on — **only FP8 crashes**, pointing at the FP8 KV path + graph-capture/shape-
     massaging interaction (crash log: "This model has shape massaging nodes that will execute on CPU
     … graph capture feature … use with caution").
  2. **`enable_cuda_graph=0` (crash workaround) → runs stably but accuracy is uniformly ~11 pt low.**
     MMLU-Pro 0.5636 with cuda-graph off, uniform across all 8 shards (per-shard 0.556–0.585, valid
     non-empty completions, only 8 error-ish log lines) — vs FP8's near-lossless GPQA/MATH/MMLU-800
     which were all measured with cuda-graph **on**. So the 0.5636 is **not a valid accuracy number**;
     it indicates the FP8 KV path produces **degraded outputs without cuda graph**.
  **Net:** FP8 KV appears numerically correct only in the cuda-graph-on path, which crashes on long
  sequences → we currently have **no reliable full MMLU-Pro for FP8**. Action: debug the FP8 GQA KV
  quant/dequant + RoPE-append kernels for (a) the SIGBUS under graph capture and (b) the cuda-graph-off
  correctness gap; then re-run FP8 MMLU-Pro with `enable_cuda_graph=1`. Until fixed, prefer **INT8-fr**.
  Repro: `MODEL_DIR=~/gpt_oss_rc_models/rc7_v2_fp8_kv_percentile RUN_DIR=/tmp/gpt_oss_rc1_runs/v2_fp8
  REQUIRE_IDLE_GPU=0 EVAL_GPUS="0..7" MMLU_PRO_MAX_SAMPLES=0 [GENAI_ENABLE_CUDA_GRAPH=0]
  run_gpt_oss_rc1_v2.sh --mmlu-pro`.
- **[RESOLVED 2026-07-12] Hardened the par-eval resume/merge** (`dev/scripts/h200_18/run_evals_parallel.sh`):
  root cause was that `oaieval`'s `LocalRecorder` opens the record file in `"wb"` (truncate) mode, so
  every resume attempt overwrote the prior partial record. Combined with a `continue`-on-success in the
  retry loop that never advanced the attempt counter, an incomplete shard could spin forever. Fix:
  (1) each attempt now writes to a private scratch record and a new `merge_records` helper folds its
  `match` rows onto the accumulated canonical record via a temp-file rename (ordered append — sample_id
  restarts at 0 per run, so dedup-by-id would be wrong), making `complete` monotonic; (2) the retry loop
  now always increments `attempt` (hard cap = `--retries + 1` rounds) and breaks early when a full round
  grades nothing new (poison-sample guard). Verified: merge preserves progress across crashes, and the
  loop terminates in the converging / poison / slow cases.

---

## 14. Leaderboard evals (MMLU-Pro / GPQA-diamond / MATH-500) for v2 prepack RCs (2026-07-12)

Full leaderboard-style accuracy for the three headline models from §12 (v2 prepack + block-size
sweep): **rc4** (best MMLU), **rc6** (balanced, int4 lm_head), **rc7** (k_quant body). Complements the
plain-MMLU numbers in §12 with the harder MMLU-Pro (10-choice) plus reasoning-heavy GPQA-diamond and
free-form MATH-500.

### Stack / method
- **Models** (pre-built, no rebuild): `~/gpt_oss_rc_models/v2_rc{4,6,7}`. **ORT** 1.29
  `~/git/onnxruntime/build/cu130_bench/Release`, **onnxruntime-genai** 0.15.0-dev, **venv**
  `~/git/onnxruntime/.venv_cu130`, 8×H200 (sm_90), CUDA 13.0.
- **Runner**: `~/git/dev/scripts/h200_18/run_gpt_oss_rc1_v2.sh --mmlu-pro --gpqa --math`, sharded
  8-way (`run_evals_parallel.sh`, GPUs 0–7). `enable_cuda_graph=1`, `strict_mode=0`, `ORT_ENABLE_XQA=1`.
- **Sample sets** (full, `*_MAX_SAMPLES=0`): MMLU-Pro **12,032** (`match_mmlu_pro`,
  `oss/gpt-oss-20b-mcq10`, A–J, `max_new_tokens=4096`); GPQA-diamond **198** (`match_gpqa_diamond`,
  `oss/gpt-oss-20b`); MATH-500 **500** (`math_500`, `oss/gpt-oss-20b-math`, symbolic `MathMatch`).

### Results

| rc  | recipe (`cuda/…`)                              | §12 MMLU | Size    | Decode | MMLU-Pro           | GPQA-diamond   | MATH-500       |
|-----|------------------------------------------------|---------:|--------:|-------:|--------------------|----------------|----------------|
| rc4 | `gpt-oss-20b_v2_rc4_default_prepack1_bs64.json`     | 0.8211 | 12.4 GB | 420.8 | **0.6973** (8390/12032) | **0.5909** (117/198) | **0.8940** (447/500) |
| rc6 | `gpt-oss-20b_v2_rc6_default_prepack1_bs64_lm4.json` | 0.8190 | 12.1 GB | 431.2 | 0.6970 (8386/12032) | 0.5657 (112/198) | 0.8920 (446/500) |
| rc7 | `gpt-oss-20b_v2_rc7_kquant_prepack0_bs64.json`      | 0.8210 | 12.4 GB | 387.3 | 0.6969 (8385/12032) | 0.5303 (105/198) | 0.8500 (425/500) |

### Observations
- **MMLU-Pro is a dead heat** (0.6973 / 0.6970 / 0.6969 — within 5 questions across 12,032). The §12
  MMLU separation (rc4 0.8211 vs rc6 0.8190) does not show up on the harder 10-choice set; all three
  land at ~0.697, ~12 pts below their plain-MMLU scores (expected — MMLU-Pro is deliberately harder).
- **rc4 leads GPQA + MATH**: GPQA-diamond 0.5909 vs rc6 0.5657 vs rc7 0.5303, and MATH-500 0.8940 vs
  0.8920 vs 0.8500. GPQA/MATH (198/500 samples) are noisier than MMLU-Pro, but the ordering is
  consistent with §12: **rc4 ≥ rc6 > rc7**.
- **rc7 (k_quant body) is the weakest across the board** — GPQA −6 pts, MATH −4.4 pts vs rc4 — echoing
  §12.2's "k_quant body does not stack on QMoE bs64" (no accuracy gain) while it also has the slowest
  decode and can't be prepacked. **rc4 dominates rc7.**
- **rc4 vs rc6**: rc6 trades ~2.5 pts GPQA and ~0.2 pt MATH for +2.5% decode (431 vs 421) and −0.3 GB;
  MMLU-Pro/MATH are effectively tied. rc4 remains the accuracy pick, rc6 the balanced pick — same
  verdict as §12.
- Grading note: the `401 Incorrect API key: dummy` lines in the shard logs are benign (evals registry
  builds an `OpenAI()` client at import); all grading is the in-process `ort_genai` completion fn.

Artifacts: `/tmp/gpt_oss_rc1_runs/v2_rc{4,6,7}_leaderboard/{scores,mmlu_pro,gpqa,math}.summary` and
per-eval `…_eval/summary_report.md`. Repro:
`MODEL_DIR=~/gpt_oss_rc_models/v2_rc4 RUN_DIR=/tmp/gpt_oss_rc1_runs/v2_rc4_leaderboard
MMLU_PRO_MAX_SAMPLES=0 GPQA_MAX_SAMPLES=0 MATH_MAX_SAMPLES=0 EVAL_GPUS="0..7" REQUIRE_IDLE_GPU=0
GENAI_ENABLE_CUDA_GRAPH=1 run_gpt_oss_rc1_v2.sh --mmlu-pro --gpqa --math` (repeat for rc6/rc7).

---

## 15. INT8 per-channel KV cache candidate on v2_rc6 — `v2_rc6_int8_kv` (2026-07-13)

Applies the §13 **INT8 per-channel KV-cache quantization** on top of the **v2_rc6** weights
(`default_prepack1_bs64_lm4`, the balanced §14 pick — int4 body + int4 lm_head, QMoE bs64,
prepacked), producing a new candidate **`v2_rc6_int8_kv`**. This differs from §13, whose base was
`rc7_default_prepack` (k_quant body); here we isolate KV quant against the **v2_rc6** weights so the
comparison is directly against the §14 leaderboard baseline. Only the KV-cache element type changes
(`present.*.key/value` → `int8`, `elem_type=3`); the int4 weights are byte-identical
(`model.onnx.data` = 12,117,786,624 B vs v2_rc6's 12,117,721,088 B — the 64 KB delta is the embedded
per-channel scale metadata, not weights). On-disk size is 12 GB for both.

Stack: 8×H200 (sm_90), ORT 1.29 (`build/cu130_bench/Release`,
`USE_FPA_INTB_GEMM/USE_FP8_KV_CACHE/USE_INT4_KV_CACHE=ON`), onnxruntime-genai
`tlwu/target_logprobs` (built Jul-12, includes the `tlwu/quantized_kv_cache` runtime dispatch —
no rebuild needed), CUDA 13.0 / cuDNN 9.19, venv `~/git/onnxruntime/.venv_cu130`.

### 15.1 Calibration & build

- **Config** (the §13 best-INT8 recipe): per-channel, `METHOD=percentile PERCENTILE=99.99`,
  **full signed range `QMAX=128`** (`scale = amax/128`, clamp `[-128,127]`), `TARGET_SEQ=512`,
  `NUM_SEQS=24`. Scales calibrated from the **v2_rc6** FP16-KV `present.*.key/value` outputs.
- Calibration clipped **98.2 %** of K channels and **95.2 %** of V channels vs raw amax (percentile
  tail trim). Scale file: `~/gpt_oss_kv_scales/kv_scales_int8_per_channel_v2_rc6.json`
  (k range [5.2e-4, 1.314] median 0.0345; v range [0.0135, 0.480] median 0.0685).
- Generated recipe: `cuda/gpt-oss-20b_v2_rc6_default_prepack1_bs64_lm4_int8_kv_gen.json`
  (= the v2_rc6 recipe + `kv_cache_quant_type=int8_per_channel` + `kv_cache_scale_file`). Model
  builder finished in 270 s. Built model: `~/gpt_oss_rc_models/v2_rc6_int8_kv`.

Command (calibrate + build + verify + benchmark):

```bash
CUDNN_HOME=/home/tianlei/cudnn9.19_cuda13 \
QUANT=int8 GRAN=per_channel METHOD=percentile PERCENTILE=99.99 TARGET_SEQ=512 NUM_SEQS=24 \
BASE_RECIPE=~/git/olive-recipes/gpt-oss-20b/cuda/gpt-oss-20b_v2_rc6_default_prepack1_bs64_lm4.json \
BASELINE_MODEL=~/gpt_oss_rc_models/v2_rc6 \
SCALE_FILE=~/gpt_oss_kv_scales/kv_scales_int8_per_channel_v2_rc6.json \
MODEL_DIR=~/gpt_oss_rc_models/v2_rc6_int8_kv \
RECIPE_OUT=~/git/olive-recipes/gpt-oss-20b/cuda/gpt-oss-20b_v2_rc6_default_prepack1_bs64_lm4_int8_kv_gen.json \
OLIVE_OUTPUT_DIR=~/git/olive-recipes/gpt-oss-20b/cuda/model_v2_rc6_int8_kv_gen \
RUN_DIR=/tmp/gpt_oss_rc1_runs/v2_rc6_int8_kv \
  dev/scripts/h200_18/run_gpt_oss_quantized_kv.sh --calibrate --build-model --verify --benchmark
```

### 15.2 Results vs baseline

Benchmark: `benchmark_e2e.py`, batch 1 / prompt 512 / gen 128, reps 5 / warmup 2, GPU0,
`enable_cuda_graph=1`, `ORT_ENABLE_XQA=1` — **both models measured with identical settings** (the
§14 decode 431.2 was a separate run; the apples-to-apples baseline re-measured here is 429.1).
Evals: `run_gpt_oss_rc1_v2.sh --mmlu-pro --gpqa --math`, full sets, 8-GPU sharded, cuda graph on.

| model | KV cache | Prefill tps | Decode tps | GPQA-diamond (198) | MATH-500 | **MMLU-Pro (12,032)** |
|-------|----------|--------:|-------:|-----:|-----:|-----:|
| **v2_rc6** (baseline) | FP16 | 15,071 | 429.1 | 0.5657 (112) | 0.8920 (446) | **0.6970** (8386) |
| **v2_rc6_int8_kv**    | INT8 per-ch | 15,359 | 355.9 | 0.6162 (122) | 0.8980 (449) | **0.6994** (8415) |
| Δ (int8 − base) |  | **+1.9 %** | **−17.1 %** | +0.0505 (+10) | +0.0060 (+3) | **+0.0024 (+29)** |

### 15.3 Findings

- **INT8 per-channel KV is essentially lossless on v2_rc6.** MMLU-Pro +0.24 pt (0.6994 vs 0.6970,
  +29/12,032 — well within noise), MATH-500 +0.6 pt (both ~0.89), GPQA-diamond +5.05 pt (0.6162 vs
  0.5657, +10/198 — favorable but 198-sample noise, ±3.5 % 1σ). Every metric is ≥ baseline. This
  reproduces §13.4's headline (INT8-fr ≈ FP16) now against the stronger v2_rc6 weights.
- **KV memory halved** (2 → 1 byte/elem in the runtime cache) at **no accuracy cost**. On-disk size
  is unchanged (12 GB) — KV quant only shrinks the GPU KV cache during inference, not stored weights.
- **Throughput**: prefill ~equal/slightly higher (+1.9 %); decode drops to 355.9 tps (−17 %) from the
  int8 dequant-in-the-GQA-kernel overhead — the same ~350-tps decode ceiling all §13 quantized-KV
  variants hit vs the FP16 baseline. The decode cost buys 2× KV-cache capacity (longer context / more
  concurrent sequences before OOM), so the trade favours long-context / high-batch serving.
- **Recommendation: `v2_rc6_int8_kv` is a viable candidate** — accuracy-neutral vs v2_rc6, half the KV
  memory, fully stable with cuda graph on (unlike the FP8 path in §13.5). Prefer it when KV-cache
  memory (context length or batch) is the binding constraint; keep FP16 v2_rc6 when raw decode tps
  is the only objective.

### 15.4 Artifacts & repro

- Model: `~/gpt_oss_rc_models/v2_rc6_int8_kv` · scales:
  `~/gpt_oss_kv_scales/kv_scales_int8_per_channel_v2_rc6.json` · recipe:
  `cuda/gpt-oss-20b_v2_rc6_default_prepack1_bs64_lm4_int8_kv_gen.json`.
- Build/benchmark logs: `/tmp/gpt_oss_rc1_runs/v2_rc6_int8_kv/` (baseline bench:
  `/tmp/gpt_oss_rc1_runs/v2_rc6_baseline_bench/benchmark.summary`).
- Eval logs/scores: `/tmp/gpt_oss_rc1_runs/v2_rc6_int8_kv_leaderboard/{scores,mmlu_pro,gpqa,math}.summary`.
- Eval repro:

```bash
CUDNN_HOME=/home/tianlei/cudnn9.19_cuda13 \
RECIPE=~/git/olive-recipes/gpt-oss-20b/cuda/gpt-oss-20b_v2_rc6_default_prepack1_bs64_lm4_int8_kv_gen.json \
MODEL_DIR=~/gpt_oss_rc_models/v2_rc6_int8_kv \
RUN_DIR=/tmp/gpt_oss_rc1_runs/v2_rc6_int8_kv_leaderboard \
MMLU_PRO_MAX_SAMPLES=0 GPQA_MAX_SAMPLES=0 MATH_MAX_SAMPLES=0 EVAL_GPUS="0..7" REQUIRE_IDLE_GPU=0 \
GENAI_ENABLE_CUDA_GRAPH=1 CLEAN_OLIVE=0 \
  dev/scripts/h200_18/run_gpt_oss_rc1_v2.sh --prepare-data --mmlu-pro --gpqa --math
```
