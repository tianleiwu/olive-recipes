# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Olive pass (and standalone CLI) that calibrates symmetric KV-cache scales.

It runs an FP16-KV baseline ONNX model over a set of calibration sequences, captures
the ``present.<layer>.key`` / ``present.<layer>.value`` outputs (post-RoPE K and raw V,
exactly the tensors the quantized ``GroupQueryAttention`` node quantizes), and computes
per-channel (or per-tensor) symmetric scales:

    scale = threshold / qmax     (qmax = 128 for INT8, 8 for INT4, 448 for FP8 E4M3)

where ``threshold`` per channel is either the abs-max (``method=minmax``) or a high
percentile of ``|x|`` (``method=percentile``, tames outliers). The result is a JSON file
consumable by the onnxruntime-genai model builder via
``extra_options["kv_cache_scale_file"]``:

    {"scales": {"k_scales": [<per-layer>...], "v_scales": [<per-layer>...]}}

As an Olive pass it takes the FP16-KV baseline (an ``ONNXModelHandler`` produced by a
preceding ``ModelBuilder`` pass), writes the scale file to ``scale_output_path``, and
returns an ``HfModelHandler`` for the original HuggingFace model so the following
``ModelBuilder`` pass can build the quantized-KV model that references the scale file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

# Diverse, long-form corpus spanning MMLU-ish domains (STEM, humanities, social science,
# reasoning). Passages are concatenated and sliced to TARGET_SEQ tokens so calibration covers
# the same post-RoPE position range that the eval context length exercises.
CORPUS = [
    "The theory of general relativity, formulated by Albert Einstein in 1915, describes gravity "
    "not as a force but as the curvature of spacetime caused by mass and energy. Massive objects "
    "such as stars and planets warp the geometry of spacetime, and this curvature dictates how "
    "other objects move. The theory predicted phenomena such as the bending of light by gravity, "
    "the precession of Mercury's orbit, gravitational time dilation, and the existence of black "
    "holes and gravitational waves, all of which have since been confirmed experimentally.",
    "In computer science, a hash table is a data structure that implements an associative array, "
    "mapping keys to values using a hash function to compute an index into an array of buckets. "
    "Collisions, where two keys hash to the same bucket, are resolved by chaining or open "
    "addressing. Under reasonable assumptions the average cost of lookup, insertion, and deletion "
    "is constant time, which makes hash tables one of the most widely used structures in practice, "
    "underpinning database indexes, caches, symbol tables in compilers, and set membership tests.",
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in two stages: the "
    "light-dependent reactions in the thylakoid membranes, which capture energy and produce ATP "
    "and NADPH, and the Calvin cycle in the stroma, which fixes carbon dioxide into sugars. "
    "Chlorophyll absorbs light most strongly in the blue and red parts of the spectrum, reflecting "
    "green light, which is why leaves appear green to the human eye under normal daylight.",
    "The French Revolution began in 1789 and led to the end of the absolute monarchy, the rise of "
    "radical political factions such as the Jacobins, the Reign of Terror, and ultimately the "
    "ascent of Napoleon Bonaparte. It was driven by financial crisis, Enlightenment ideas about "
    "popular sovereignty and natural rights, and deep resentment of aristocratic privilege. The "
    "Declaration of the Rights of Man proclaimed liberty, equality, and fraternity, principles that "
    "reshaped European politics and inspired revolutionary and nationalist movements worldwide.",
    "Quantum mechanics is the branch of physics that studies matter and energy at the scale of "
    "atoms and subatomic particles, where classical intuitions break down. Particles exhibit both "
    "wave and particle behavior, quantities such as energy are quantized, and the act of "
    "measurement affects the system. The Heisenberg uncertainty principle sets a fundamental limit "
    "on the simultaneous knowledge of position and momentum, while the Schrodinger equation "
    "governs how the wavefunction, encoding the probabilities of outcomes, evolves over time.",
    "In economics, supply and demand describe how prices are determined in a competitive market. "
    "The demand curve slopes downward because consumers buy more of a good at lower prices, while "
    "the supply curve slopes upward because producers supply more at higher prices. The equilibrium "
    "price occurs where the two curves intersect, clearing the market. Shifts in demand or supply, "
    "caused by changes in income, preferences, technology, or input costs, move the equilibrium and "
    "explain phenomena such as shortages, surpluses, and the effects of taxes and price controls.",
    "The human circulatory system transports oxygen, nutrients, hormones, and waste products "
    "throughout the body. The heart, a muscular pump, drives blood through arteries, capillaries, "
    "and veins. Oxygenated blood leaves the left ventricle through the aorta to the tissues, while "
    "deoxygenated blood returns to the right side of the heart and is pumped to the lungs for gas "
    "exchange. Red blood cells carry oxygen bound to hemoglobin, and the coordinated contraction of "
    "cardiac muscle is regulated by electrical signals originating in the sinoatrial node.",
    "Machine learning models learn patterns from data by minimizing a loss function through "
    "iterative optimization such as gradient descent, generalizing from training examples to unseen "
    "inputs. Overfitting occurs when a model memorizes noise in the training set and fails to "
    "generalize; it is mitigated by regularization, early stopping, dropout, and larger or more "
    "diverse datasets. The bias-variance tradeoff captures the tension between models that are too "
    "simple to fit the data and models so flexible that they are sensitive to sampling noise.",
    "In organic chemistry, the carbon atom's ability to form four covalent bonds allows it to build "
    "long chains, branched structures, and rings, giving rise to the vast diversity of organic "
    "molecules. Functional groups such as hydroxyl, carbonyl, carboxyl, and amino groups determine "
    "the chemical reactivity and physical properties of compounds. Isomers share a molecular formula "
    "but differ in connectivity or spatial arrangement, and stereochemistry, including chirality, "
    "profoundly affects how molecules interact with biological systems such as enzymes and receptors.",
    "The United States Constitution, ratified in 1788, establishes a federal system that divides "
    "power between the national government and the states, and separates the national government "
    "into legislative, executive, and judicial branches. A system of checks and balances lets each "
    "branch limit the others: Congress writes laws and controls spending, the President enforces "
    "laws and commands the military, and the courts interpret laws and can strike down those that "
    "violate the Constitution. The Bill of Rights guarantees fundamental liberties such as speech.",
    "Plate tectonics is the theory that Earth's rigid outer shell, the lithosphere, is divided into "
    "plates that move slowly over the more fluid asthenosphere beneath. At divergent boundaries "
    "plates pull apart and new crust forms; at convergent boundaries plates collide, producing "
    "mountains, volcanoes, and deep ocean trenches through subduction; and at transform boundaries "
    "plates slide past one another, generating earthquakes. This unifying theory explains the "
    "distribution of continents, the pattern of seismic activity, and the geologic history of Earth.",
    "In probability theory, the central limit theorem states that the sum or average of a large "
    "number of independent, identically distributed random variables tends toward a normal "
    "distribution, regardless of the underlying distribution, provided the variance is finite. This "
    "result explains why the bell curve appears so often in nature and underpins much of statistical "
    "inference, including confidence intervals and hypothesis tests. The standard deviation of the "
    "sample mean shrinks in proportion to the inverse square root of the sample size as data grows.",
    "The Industrial Revolution, beginning in Britain in the late eighteenth century, transformed "
    "economies from agrarian and handcraft production to machine-based manufacturing. Innovations "
    "such as the steam engine, the power loom, and improvements in iron and steel production raised "
    "productivity dramatically. Urbanization accelerated as workers moved to factory towns, living "
    "standards eventually rose, and new social classes emerged. The period also brought harsh labor "
    "conditions, child labor, and pollution, prompting reform movements, labor unions, and new laws.",
    "In cell biology, mitochondria are membrane-bound organelles that generate most of the cell's "
    "supply of adenosine triphosphate, used as chemical energy. Through oxidative phosphorylation, "
    "electrons are passed along a chain of protein complexes in the inner membrane, pumping protons "
    "to create a gradient that drives ATP synthase. Mitochondria possess their own circular DNA and "
    "are thought to have originated from an ancient endosymbiotic bacterium, a hypothesis supported "
    "by their double membrane, independent replication, and similarities to modern prokaryotes.",
    "Linguistics is the scientific study of language and its structure, encompassing phonetics, "
    "phonology, morphology, syntax, semantics, and pragmatics. Phonology studies sound systems, "
    "morphology the structure of words, syntax the rules that combine words into sentences, and "
    "semantics the meaning conveyed. Languages change over time through sound shifts, borrowing, and "
    "grammaticalization, and comparative methods reconstruct ancestral languages. Chomsky's theory "
    "of universal grammar proposed that humans share an innate capacity underlying all languages.",
    "The greenhouse effect is the process by which certain gases in a planet's atmosphere, such as "
    "carbon dioxide, methane, and water vapor, trap heat by absorbing and re-emitting infrared "
    "radiation. This natural effect keeps Earth warm enough to support life, but human activities "
    "since the Industrial Revolution have increased greenhouse gas concentrations, enhancing the "
    "effect and driving global warming. Consequences include rising sea levels, more frequent "
    "extreme weather, ocean acidification, and shifts in ecosystems and agricultural patterns.",
]

# quant_type -> scale divisor qmax = 2^(bits-1) for signed int (128 int8, 8 int4); 448 for fp8 E4M3.
_QMAX = {"int8": 128.0, "int4": 8.0, "fp8": 448.0}


def _bits_key(quant_type: str) -> str:
    for key in _QMAX:
        if quant_type.startswith(key):
            return key
    raise ValueError(f"Unsupported kv_cache quant_type '{quant_type}' (expect int8/int4/fp8 prefix).")


def _tokenize_corpus(tokenizer, num_seqs: int, target_seq: int) -> list[np.ndarray]:
    """Build ``num_seqs`` calibration sequences of length ``target_seq`` tokens.

    Concatenates the corpus into one long token stream and slices non-overlapping windows so
    every window covers positions 0..target_seq-1 (matching the eval context length), exercising
    the full post-RoPE position range rather than only the first few dozen positions.
    """
    ids: list[int] = []
    joined = "\n\n".join(CORPUS)
    while len(ids) < num_seqs * target_seq + target_seq:
        ids.extend(tokenizer.encode(joined))
    seqs = []
    for s in range(num_seqs):
        window = ids[s * target_seq : (s + 1) * target_seq]
        if len(window) < target_seq:
            break
        seqs.append(np.asarray(window, dtype=np.int64).reshape(1, target_seq))
    return seqs


def calibrate_kv_scales(
    model_path: str,
    tokenizer_path: str,
    out_json: str,
    quant_type: str = "int8_per_channel",
    method: str = "percentile",
    percentile: float = 99.99,
    target_seq: int = 512,
    num_seqs: int = 24,
    num_layers: int | None = None,
    num_kv_heads: int = 8,
    head_size: int = 64,
) -> str:
    """Run the FP16-KV baseline and write calibrated symmetric KV scales to ``out_json``.

    Returns the path to the written scale file.
    """
    import onnxruntime as ort  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    quant_type = quant_type.lower()
    qmax = _QMAX[_bits_key(quant_type)]
    per_channel = quant_type.endswith("per_channel")
    is_fp8 = quant_type.startswith("fp8")
    # Full-range signed int: quantizer clamps to [-qmax, qmax-1] (kInt8 [-128,127] / kInt4 [-8,7]).
    qneg, qpos = (-qmax, qmax) if is_fp8 else (-qmax, qmax - 1.0)

    # Prepacked MatMulNBits baselines require the fpA_intB path to load; harmless otherwise.
    os.environ.setdefault("ORT_FPA_INTB_GEMM", "1")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        model_path, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    output_names = [o.name for o in sess.get_outputs()]
    present_keys = sorted(
        (n for n in output_names if n.startswith("present.") and n.endswith(".key")),
        key=lambda n: int(n.split(".")[1]),
    )
    if num_layers is None:
        num_layers = len(present_keys)
    if num_layers == 0:
        raise ValueError("Model has no present.*.key outputs; cannot calibrate KV scales.")
    read_names = [f"present.{i}.key" for i in range(num_layers)] + [
        f"present.{i}.value" for i in range(num_layers)
    ]
    channels = num_kv_heads * head_size

    past_names = [i.name for i in sess.get_inputs() if i.name.startswith("past_key_values.")]
    seqs = _tokenize_corpus(tokenizer, num_seqs, target_seq)
    logger.info(
        "KV calibration: %d sequences x %d tokens (quant=%s method=%s qmax=%g)",
        len(seqs),
        target_seq,
        quant_type,
        method,
        qmax,
    )

    k_amax = np.zeros((num_layers, channels), dtype=np.float32)
    v_amax = np.zeros((num_layers, channels), dtype=np.float32)
    need_samples = method == "percentile"
    token_budget = 12288
    per_layer_budget = max(1, token_budget // max(1, len(seqs)))
    k_buf = [[] for _ in range(num_layers)] if need_samples else None
    v_buf = [[] for _ in range(num_layers)] if need_samples else None

    for pi, input_ids in enumerate(seqs):
        seq_len = input_ids.shape[1]
        feeds = {
            "input_ids": input_ids,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        }
        for name in past_names:
            feeds[name] = np.zeros((1, num_kv_heads, 0, head_size), dtype=np.float16)
        outputs = sess.run(read_names, feeds)

        for i in range(num_layers):
            k = outputs[i].astype(np.float32).reshape(num_kv_heads, seq_len, head_size)
            v = outputs[num_layers + i].astype(np.float32).reshape(num_kv_heads, seq_len, head_size)
            k = np.transpose(k, (1, 0, 2)).reshape(seq_len, channels)
            v = np.transpose(v, (1, 0, 2)).reshape(seq_len, channels)
            k_amax[i] = np.maximum(k_amax[i], np.abs(k).max(axis=0))
            v_amax[i] = np.maximum(v_amax[i], np.abs(v).max(axis=0))
            if need_samples:
                take = min(per_layer_budget, seq_len)
                k_buf[i].append(k[:take].astype(np.float16))
                v_buf[i].append(v[:take].astype(np.float16))
        logger.info("[%d/%d] seq_len=%d k_amax<L0>=%.4f v_amax<L0>=%.4f", pi + 1, len(seqs), seq_len, k_amax[0].max(), v_amax[0].max())

    k_thr = k_amax.copy()
    v_thr = v_amax.copy()
    if method == "percentile":
        for i in range(num_layers):
            ks = np.concatenate(k_buf[i], axis=0).astype(np.float32)
            vs = np.concatenate(v_buf[i], axis=0).astype(np.float32)
            k_thr[i] = np.percentile(np.abs(ks), percentile, axis=0)
            v_thr[i] = np.percentile(np.abs(vs), percentile, axis=0)
    elif method != "minmax":
        raise ValueError(f"Unknown method={method!r} (minmax|percentile)")

    k_clip = float(np.mean(k_amax > k_thr * 1.0001))
    v_clip = float(np.mean(v_amax > v_thr * 1.0001))

    if not per_channel:
        k_thr = k_thr.max(axis=1, keepdims=True)
        v_thr = v_thr.max(axis=1, keepdims=True)

    k_scales = np.maximum(k_thr, 1e-6) / qmax
    v_scales = np.maximum(v_thr, 1e-6) / qmax

    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"scales": {"k_scales": k_scales.tolist(), "v_scales": v_scales.tolist()}}, f)

    # qneg is only meaningful when reproducing quant/dequant; keep referenced for clarity.
    del qneg, qpos
    logger.info(
        "Wrote %s (quant=%s, per_channel=%s). k clipped=%.3f v clipped=%.3f k_scale[%.6f,%.6f] v_scale[%.6f,%.6f]",
        out_json,
        quant_type,
        per_channel,
        k_clip,
        v_clip,
        float(k_scales.min()),
        float(k_scales.max()),
        float(v_scales.min()),
        float(v_scales.max()),
    )
    return out_json


# --------------------------------------------------------------------------- #
# Olive pass wrapper
# --------------------------------------------------------------------------- #
if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ONNXModelHandler
    from olive.passes.pass_config import BasePassConfig

try:
    from olive.model import HfModelHandler
    from olive.passes import Pass
    from olive.passes.olive_pass import PassConfigParam

    _OLIVE_AVAILABLE = True
except ImportError:  # pragma: no cover - allows standalone CLI use without olive installed
    _OLIVE_AVAILABLE = False


if _OLIVE_AVAILABLE:

    class KvCacheScaleCalibration(Pass):
        """Calibrate symmetric KV-cache scales from an FP16-KV baseline ONNX model.

        Input: the FP16-KV baseline (``ONNXModelHandler``) from a preceding ``ModelBuilder`` pass.
        Effect: writes a ``kv_cache_scale_file`` JSON to ``scale_output_path``.
        Output: an ``HfModelHandler`` for ``hf_model_path`` so the next ``ModelBuilder`` pass can
        build the quantized-KV model referencing the scale file.
        """

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
                "num_seqs": PassConfigParam(
                    type_=int, default_value=24, description="Number of calibration sequences."
                ),
                "num_layers": PassConfigParam(
                    type_=int, default_value=None, description="Layer count (auto-detected from model if omitted)."
                ),
                "num_kv_heads": PassConfigParam(type_=int, default_value=8, description="KV heads."),
                "head_size": PassConfigParam(type_=int, default_value=64, description="Attention head size."),
            }

        def _run_for_config(
            self, model: "ONNXModelHandler", config: type["BasePassConfig"], output_model_path: str
        ) -> "HfModelHandler":
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
            )
            return HfModelHandler(model_path=config.hf_model_path, load_kwargs=config.hf_load_kwargs)


def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Calibrate symmetric KV-cache scales from an FP16-KV ONNX model.")
    p.add_argument("--model", required=True, help="Path to the FP16-KV baseline model.onnx (or its directory).")
    p.add_argument("--tokenizer", required=True, help="HF model id/path for the tokenizer.")
    p.add_argument("--out", required=True, help="Output scale JSON path.")
    p.add_argument("--quant-type", default="int8_per_channel")
    p.add_argument("--method", default="percentile", choices=["minmax", "percentile"])
    p.add_argument("--percentile", type=float, default=99.99)
    p.add_argument("--target-seq", type=int, default=512)
    p.add_argument("--num-seqs", type=int, default=24)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-size", type=int, default=64)
    args = p.parse_args()

    model_path = args.model
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.onnx")
    calibrate_kv_scales(
        model_path=model_path,
        tokenizer_path=args.tokenizer,
        out_json=args.out,
        quant_type=args.quant_type,
        method=args.method,
        percentile=args.percentile,
        target_seq=args.target_seq,
        num_seqs=args.num_seqs,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
    )


if __name__ == "__main__":
    _main()
