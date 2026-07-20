# Qwen3.6-35B-A3B INT4 / INT8 (CUDA)

Export [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (the **bf16**
base checkpoint) to an ONNX Runtime GenAI model, quantizing weights to **int4 / int8** with
the ONNX Runtime GenAI model builder (RTN weight-only quantization).

These recipes are the **integer** counterpart to the sibling
[`Qwen-Qwen3.6-35B-A3B-NVFP4`](../../Qwen-Qwen3.6-35B-A3B-NVFP4/cuda/) recipes. On GPUs without
native FP4 (e.g. **H200 / Hopper**), int4 / int8 weight-only quantization can decode faster than
the NVFP4 path (which runs through a dequant fallback on SM90), so these candidates exist to
compare accuracy (MMLU-Pro) and throughput (prefill / decode / MTP TPS) against NVFP4.

Unlike NVFP4 (which reads pre-quantized experts from the `nvidia/Qwen3.6-35B-A3B-NVFP4`
checkpoint), these recipes quantize the **raw bf16 experts** on the fly, so the input model is
the original `Qwen/Qwen3.6-35B-A3B`.

## Quantization scheme

All three recipes share the same layout, differing only in the MoE quantization:

- **MoE experts (routed + shared)**: `com.microsoft.QMoE`, block-wise weight-only quant
  (`moe_quant_type` + `qmoe_block_size`).
- **Dense MatMul body** (attention q/k/v/o, router, gates): int4 `MatMulNBits`, block size 32,
  `int4_algo_config=default` (MLAS).
- **Mixed-precision int8 MatMuls** (`matmul_mixed_precision=last_matmul:int8,mixed_layers:int8`):
  - `last_matmul:int8` → the `lm_head` logits GEMM in int8.
  - `mixed_layers:int8` → the accuracy-sensitive attention layers (first 1/8 + last 1/8 + every
    3rd layer) in int8. This is the same placement found useful on GPT-OSS-20B.
- **Embedding**: kept in fp16 (`exclude_embeds=false`, io dtype fp16).
- **MTP self-speculative head**: single-layer int8 `QMoE` (`enable_mtp`,
  `mtp_head_quant_type=int8`), emitted alongside the decoder in the same build (`text.onnx` +
  `mtp.onnx`), with `hidden_states` and CUDA-graph metadata.

## Recipes

| Config | Output | MoE quant | Notes |
|--------|--------|-----------|-------|
| `qwen3.6-35b-a3b-int4-b32_cuda.json` | `cuda/models/int4-b32/{text.onnx,mtp.onnx}` | int4, block 32 | Primary candidate (finest MoE granularity). |
| `qwen3.6-35b-a3b-int4-b64_cuda.json` | `cuda/models/int4-b64/{text.onnx,mtp.onnx}` | int4, block 64 | Smaller / faster MoE. |
| `qwen3.6-35b-a3b-int4-b128_cuda.json` | `cuda/models/int4-b128/{text.onnx,mtp.onnx}` | int4, block 128 | Smallest / fastest MoE (coarsest granularity). |
| `qwen3.6-35b-a3b-int8moe_cuda.json`  | `cuda/models/int8moe/{text.onnx,mtp.onnx}`  | int8, block 32 | Accuracy reference (larger). |

Each recipe produces both `text.onnx` (used for MMLU-Pro and plain-decode benchmarking) and
`mtp.onnx` (used for MTP self-speculative decode benchmarking) in a **single** `ModelBuilder`
pass — the MTP head is emitted when `enable_mtp=true`.

## Build

```bash
# Primary int4 candidate (MoE block 32)
olive run --config qwen3.6-35b-a3b-int4-b32_cuda.json

# int4 MoE block 64
olive run --config qwen3.6-35b-a3b-int4-b64_cuda.json

# int8 MoE accuracy reference
olive run --config qwen3.6-35b-a3b-int8moe_cuda.json
```

## Requirements

- An ONNX Runtime CUDA build (int4/int8 `MatMulNBits` + `QMoE` are standard; no FP4 build
  required — but a build with `onnxruntime_USE_FP4_QMOE=ON` also works for these recipes).
- `onnxruntime-genai` built from a branch that includes the `moe_quant_type` /
  `matmul_mixed_precision` / MTP model-builder support. The **source** builder must be used — the
  released wheel predates these options. Overlay it into the environment once:

  ```bash
  PKG=$(python -c "import onnxruntime_genai, os; print(os.path.dirname(onnxruntime_genai.__file__))")
  cp <onnxruntime-genai>/src/python/py/models/builder.py    "$PKG/models/"
  cp -r <onnxruntime-genai>/src/python/py/models/builders/. "$PKG/models/builders/"
  find "$PKG/models" -name __pycache__ -type d -prune -exec rm -rf {} +
  ```

- `transformers==5.14.1` (or newer) is required to load the `qwen3_5_moe` checkpoint.
- A GPU (the MTP recipe emits CUDA-graph metadata, `enable_cuda_graph=1`).

See `dev/docs/memory/qwen_3.6_nvfp4_quick_start_v2.md` for the shared build environment and the
overlay procedure.
