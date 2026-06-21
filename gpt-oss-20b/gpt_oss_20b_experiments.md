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

