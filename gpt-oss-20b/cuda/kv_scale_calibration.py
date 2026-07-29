# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Olive pass wrapper around the onnxruntime-genai KV-cache scale calibrator.

The calibration itself is a general model-builder tool and now lives in
onnxruntime-genai as ``onnxruntime_genai.models.kv_cache_calibration``; this file only
adapts it to the Olive pass interface so a single ``olive run`` can do

    ModelBuilder (FP16-KV baseline) -> KvCacheScaleCalibration -> ModelBuilder (quantized KV)

The pass takes the FP16-KV baseline (an ``ONNXModelHandler`` from the preceding
``ModelBuilder`` pass), writes a ``kv_cache_scale_file`` JSON to ``scale_output_path``, and
returns an ``HfModelHandler`` for ``hf_model_path`` so the next ``ModelBuilder`` pass can
build the quantized-KV model referencing that file.

Register it once in ``<olive>/olive/olive_config.json`` and put this directory on
``PYTHONPATH`` when running the recipe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ONNXModelHandler
    from olive.passes.pass_config import BasePassConfig


def _load_calibrator():
    """Import ``calibrate_kv_scales`` from the installed onnxruntime-genai."""
    try:
        from onnxruntime_genai.models.kv_cache_calibration import calibrate_kv_scales
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import onnxruntime_genai.models.kv_cache_calibration. Install an "
            "onnxruntime-genai build that ships the KV-cache calibration tool, or add "
            "<onnxruntime-genai>/src/python/py to PYTHONPATH."
        ) from exc
    return calibrate_kv_scales


class KvCacheScaleCalibration(Pass):
    """Calibrate symmetric KV-cache scales from an FP16-KV baseline ONNX model."""

    @classmethod
    def _default_config(cls, accelerator_spec: "AcceleratorSpec") -> dict[str, "PassConfigParam"]:
        return {
            "hf_model_path": PassConfigParam(
                type_=str,
                required=True,
                description="HuggingFace model id/path to emit as the output model for the next pass.",
            ),
            "scale_output_path": PassConfigParam(
                type_=str,
                required=True,
                description="Absolute path to write the calibrated kv_cache_scale_file JSON.",
            ),
            "hf_load_kwargs": PassConfigParam(
                type_=dict,
                default_value={"torch_dtype": "float16"},
                description="load_kwargs forwarded to the emitted HfModelHandler.",
            ),
            "quant_type": PassConfigParam(
                type_=str,
                default_value="int8_per_channel",
                description="KV quant scheme: {int8,int4,fp8}_{per_channel,per_tensor}.",
            ),
            "method": PassConfigParam(
                type_=str, default_value="percentile", description="Calibration objective: minmax | percentile."
            ),
            "percentile": PassConfigParam(
                type_=float, default_value=99.99, description="Percentile for method=percentile."
            ),
            "target_seq": PassConfigParam(
                type_=int, default_value=512, description="Calibration sequence length in tokens."
            ),
            "num_seqs": PassConfigParam(type_=int, default_value=24, description="Number of calibration sequences."),
            "num_layers": PassConfigParam(
                type_=int, default_value=None, description="Layer count (auto-detected from the model if omitted)."
            ),
            "num_kv_heads": PassConfigParam(
                type_=int, default_value=None, description="KV heads (auto-detected from the model if omitted)."
            ),
            "head_size": PassConfigParam(
                type_=int, default_value=None, description="Head size (auto-detected from the model if omitted)."
            ),
            "k_rotary_envelope": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Calibrate K on the rotation-invariant RoPE pair norm so the scales stay "
                    "valid far beyond the calibration window (required for long-CoT accuracy)."
                ),
            ),
        }

    def _run_for_config(
        self, model: "ONNXModelHandler", config: type["BasePassConfig"], output_model_path: str
    ) -> "HfModelHandler":
        calibrate_kv_scales = _load_calibrator()
        calibrate_kv_scales(
            model_path=model.model_path,
            tokenizer_path=config.hf_model_path,
            out_json=config.scale_output_path,
            quant_type=config.quant_type,
            method=config.method,
            percentile=config.percentile,
            target_seq=config.target_seq,
            num_seqs=config.num_seqs,
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_size=config.head_size,
            k_rotary_envelope=config.k_rotary_envelope,
        )
        return HfModelHandler(model_path=config.hf_model_path, load_kwargs=config.hf_load_kwargs)
