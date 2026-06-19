# GPT-OSS-20b Performance Fix - Quick Reference

## The Issue
New model variants showed **5-6x lower generation throughput** (~50-90 TPS vs expected ~300 TPS).

## The Cause
CUDA graph was disabled in `genai_config.json`:
```json
"enable_cuda_graph": "0"   // ❌ WRONG (50 TPS)
```

## The Fix
Enable CUDA graph in all model configs:
```bash
# Applied to all 4 variants:
sed -i 's/"enable_cuda_graph": "0"/"enable_cuda_graph": "1"/' \
  /home/tianlei/olive-recipes/gpt-oss-20b/cuda/variants/*/genai_config.json
```

## The Result
✅ **Generation throughput: 275 TPS** (back to baseline performance)

## Performance Comparison

| Phase | Before Fix | After Fix | Improvement |
|-------|-----------|-----------|-------------|
| INT4 Decode | 48.54 TPS | 275.4 TPS | **+467%** ✅ |
| INT8 Decode | 63.23 TPS | 269.3 TPS | **+326%** ✅ |

## Files Updated

1. **genai_config.json** (in all 4 variants)
   - Changed: `"enable_cuda_graph": "0"` → `"1"`

2. **run_gpt_oss_model_build.sh** 
   - Added auto-fix: Enables CUDA graph during post-build step
   - Future runs will automatically get correct setting

3. **Documentation**
   - [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) — Detailed root cause
   - [FIX_SUMMARY.md](FIX_SUMMARY.md) — Complete analysis and verification
   - [README_FIX.md](README_FIX.md) — This file (quick reference)

## Verification

To verify models now have CUDA graph enabled:

```bash
grep -r "enable_cuda_graph" /home/tianlei/olive-recipes/gpt-oss-20b/cuda/variants/
# Expected output: all should show "1"
```

To benchmark performance:

```bash
cd /home/tianlei/onnxruntime-genai
source /home/tianlei/venv/bin/activate

python benchmark/python/benchmark_e2e.py \
  -i /home/tianlei/olive-recipes/gpt-oss-20b/cuda/variants/cuda_int4_int4_qmoe_default \
  -e cuda -b 1 -l 512 -g 128 -r 3 -w 1 \
  --use_random_tokens --chat_template "{input}" \
  -mn gpt-oss-20b -pr int4

# Expected: ~275 TPS (Token Generation Throughput)
```

## Impact Summary

| Aspect | Status |
|--------|--------|
| Generation throughput | ✅ Restored to 275 TPS (baseline) |
| Model quality | ✅ No change (same quantization) |
| Latency | ✅ Improved (5-6x faster decoding) |
| Future builds | ✅ Auto-fixed via updated build script |

---

**Status**: ✅ RESOLVED  
**Fix Date**: 2026-06-16  
**Models Ready**: All 4 variants ready for production use  
