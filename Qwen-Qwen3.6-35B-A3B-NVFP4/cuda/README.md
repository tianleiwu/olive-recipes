# Qwen3.6-35B-A3B-NVFP4 (CUDA)

Export [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
to an ONNX Runtime GenAI model. This is the NVFP4 (4-bit) checkpoint of Qwen3.6-35B-A3B,
quantized by [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer):

- **MoE experts + shared expert**: `W4A16_NVFP4`, group size 16 (E2M1 weights, FP8-E4M3
  block scales, per-expert FP32 global scale). Exported directly to the ONNX Runtime
  `com.microsoft.QMoE` op with `quant_type="nvfp4"`, `block_size=16` (no re-quantization).
- Attention projections are FP8 in the source checkpoint.

NVFP4 MoE quantization is selected with the unified `--extra_options moe_quant_type=nvfp4`
option (same design pattern as `mxfp4` / `int8`). This replaces the legacy `use_nvfp4_moe=true`
flag; the produced ONNX weights are byte-identical.

## Recipes

| Config | Output | Description |
|--------|--------|-------------|
| `qwen3.6-35b-a3b-nvfp4_cuda.json` | `cuda/models/text.onnx` | Plain NVFP4 decoder only. |
| `qwen3.6-35b-a3b-nvfp4-mtp_cuda.json` | `cuda/models/mtp/{text.onnx,mtp.onnx}` | Decoder **plus** an int8 MTP self-speculative head and per-position recurrent/conv state outputs (`hidden_states`, `present.*.recurrent_state_all`, `present.*.conv_state_all`). |

Both `text.onnx` and `mtp.onnx` are produced by a **single** `ModelBuilder` pass — the MTP head is
emitted alongside the decoder when `enable_mtp=true`, so it does not need a separate recipe file
(unlike the vision/embedding sub-models of the base Qwen3.6-35B-A3B recipe, which use distinct
loaders). Olive returns a composite (multi-file) model for the MTP recipe.

## Build

```bash
# Plain decoder
olive run --config qwen3.6-35b-a3b-nvfp4_cuda.json

# Decoder + MTP head (+ recurrent-state outputs)
olive run --config qwen3.6-35b-a3b-nvfp4-mtp_cuda.json
```

The routed experts are read directly from the source safetensors and packed into the
QMoE NVFP4 layout by the ONNX Runtime GenAI model builder — no re-quantization.

## Requirements

- An ONNX Runtime CUDA build with FP4 QMoE enabled
  (`onnxruntime_USE_FP4_QMOE=ON`, `ENABLE_FP4`); the `quant_type="nvfp4"` QMoE runs via the
  dequant-fallback path (works on SM90/Hopper such as H200; native block-scaled FP4 GEMM is
  Blackwell-only).
- `onnxruntime-genai` built from a branch that includes the NVFP4 (`moe_quant_type`) and MTP
  model-builder support. The **source** builder must be used — the released wheel predates these
  options. Overlay it into the environment once:

  ```bash
  PKG=$(python -c "import onnxruntime_genai, os; print(os.path.dirname(onnxruntime_genai.__file__))")
  cp <onnxruntime-genai>/src/python/py/models/builder.py    "$PKG/models/"
  cp -r <onnxruntime-genai>/src/python/py/models/builders/. "$PKG/models/builders/"
  find "$PKG/models" -name __pycache__ -type d -prune -exec rm -rf {} +
  ```

- `transformers==5.14.1` (or newer) is required to load the `qwen3_5_moe` checkpoint.
- The MTP recipe requires a GPU; it emits CUDA-graph metadata (`enable_cuda_graph=1`).

See `dev/docs/memory/qwen_3.6_nvfp4_quick_start_v2.md` for the full step-by-step guide and the
byte-identity verification procedure.
