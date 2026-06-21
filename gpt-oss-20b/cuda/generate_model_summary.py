#!/usr/bin/env python3
"""Generate a consolidated per-model summary JSON for a gpt-oss-20b variant.

The backbone of the output is the benchmark-results JSON produced by
``benchmark_e2e.py`` (e.g. ``bench_<variant>.json``). On top of that backbone we
embed the extra provenance needed to reproduce / interpret a result:

  * git branch + commit for onnxruntime, onnxruntime-genai, evals
  * the full Olive recipe JSON used to build the model (embedded inline)
  * provider_options (extracted from the model's genai_config.json)
  * MMLU evaluation options (generation knobs from the evals completion fn) and
    the MMLU score
  * environment variables used in the test (e.g. kernel enable/disable toggles)

Typical usage (manual, one invocation per variant)::

    ./generate_model_summary.py \
        --variant cuda_int4_int4_qmoe_rtn_mixed_matmul_only_qknorm_bs0 \
        --mmlu-score 0.7933 \
        --env-set ORT_ENABLE_XQA=1

Most paths are derived from ``--variant`` relative to this script's directory,
but every path can be overridden explicitly. See ``--help``.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")

# Environment variables we record by default (test-time kernel / runtime toggles).
DEFAULT_ENV_VARS = [
    "ORT_ENABLE_XQA",
    "CUDA_VISIBLE_DEVICES",
]


def warn(msg: str) -> None:
    print(f"[summary][warn] {msg}", file=sys.stderr)


def git_info(repo_dir: str) -> dict:
    """Return {branch, commit, dirty, describe} for a git repo, or an error note."""
    if not repo_dir or not os.path.isdir(repo_dir):
        return {"path": repo_dir, "error": "directory not found"}

    def _run(args):
        try:
            return subprocess.check_output(
                ["git", "-C", repo_dir, *args],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    info = {
        "path": repo_dir,
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["rev-parse", "HEAD"]),
        "describe": _run(["describe", "--always", "--tags", "--dirty"]),
    }
    status = _run(["status", "--porcelain"])
    info["dirty"] = bool(status) if status is not None else None
    if info["commit"] is None:
        info["error"] = "not a git repository or git unavailable"
    return info


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_provider_options(genai_config_path: str):
    """Pull provider_options out of a model's genai_config.json."""
    if not os.path.isfile(genai_config_path):
        warn(f"genai_config.json not found: {genai_config_path}")
        return None
    cfg = load_json(genai_config_path)
    try:
        sess = cfg["model"]["decoder"]["session_options"]
    except (KeyError, TypeError):
        warn("session_options not found in genai_config.json")
        return None
    return {
        "provider_options": sess.get("provider_options"),
        "search": cfg.get("search"),
    }


def _parse_kv_list(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            warn(f"ignoring malformed key=value: {item!r}")
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _coerce(value: str):
    """Best-effort scalar coercion for CLI-provided string values."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def read_completion_fn_options(evals_dir: str, comp_fn: str) -> dict:
    """Read the configured args for a completion fn (e.g. oss/gpt-oss-20b).

    These are the MMLU generation options (max_new_tokens, do_sample, ...).
    We read them straight from the evals registry YAML so that new options added
    in future evals commits are picked up automatically once pulled.
    """
    result = {"completion_fn": comp_fn, "class": None, "args": None, "source": None}
    reg_dir = os.path.join(evals_dir, "evals", "registry", "completion_fns")
    if not os.path.isdir(reg_dir):
        warn(f"completion_fns registry not found: {reg_dir}")
        return result
    try:
        import yaml  # provided by the evals install
    except ImportError:
        warn("pyyaml not available; skipping completion-fn option extraction")
        return result

    for path in sorted(glob.glob(os.path.join(reg_dir, "*.yaml"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            warn(f"could not parse {path}: {e}")
            continue
        if isinstance(data, dict) and comp_fn in data:
            entry = data[comp_fn] or {}
            result["class"] = entry.get("class")
            result["args"] = entry.get("args")
            result["source"] = os.path.relpath(path, evals_dir)
            return result

    warn(f"completion fn {comp_fn!r} not found in {reg_dir}")
    return result


def pool_mmlu_accuracy(eval_dir: str):
    """Pool match_mmlu_shard_*.jsonl match rows in a dir into an accuracy."""
    total = correct = 0
    pattern = os.path.join(eval_dir, "**", "match_mmlu_shard_*.jsonl")
    for path in glob.glob(pattern, recursive=True):
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
    if not total:
        return None, 0
    return round(correct / total, 4), total


def mmlu_score_from_tsv(tsv_path: str, variant: str):
    """Look up the MMLU score (4th column) for a variant in the results TSV."""
    if not os.path.isfile(tsv_path):
        return None
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0] == variant and len(parts) >= 4 and parts[3].strip():
                try:
                    return float(parts[3])
                except ValueError:
                    return parts[3]
    return None


def build_summary(args) -> dict:
    backbone = load_json(args.bench_json)

    # ---- git versions ----
    versions = {
        "onnxruntime": git_info(args.ort_dir),
        "onnxruntime_genai": git_info(args.genai_dir),
        "evals": git_info(args.evals_dir),
    }

    # ---- olive recipe (embedded) ----
    recipe = {"path": args.recipe, "content": None}
    if args.recipe and os.path.isfile(args.recipe):
        recipe["content"] = load_json(args.recipe)
    else:
        warn(f"recipe not found: {args.recipe}")

    # ---- provider options ----
    provider = extract_provider_options(args.genai_config)

    # ---- MMLU options + score ----
    mmlu = read_completion_fn_options(args.evals_dir, args.completion_fn)
    # CLI overrides for generation options (e.g. runtime COMP_ARGS overrides).
    overrides = {k: _coerce(v) for k, v in _parse_kv_list(args.mmlu_option).items()}
    if overrides:
        merged = dict(mmlu.get("args") or {})
        merged.update(overrides)
        mmlu["args"] = merged
        mmlu["overrides"] = overrides
    mmlu["eval"] = args.mmlu_eval
    mmlu["max_samples"] = args.mmlu_max_samples if args.mmlu_max_samples > 0 else "full"
    mmlu["evals_commit"] = versions["evals"].get("commit")

    score = args.mmlu_score
    sample_count = None
    if score is None and args.eval_dir:
        score, sample_count = pool_mmlu_accuracy(args.eval_dir)
        if sample_count:
            mmlu["sample_count"] = sample_count
    if score is None and args.results_tsv:
        score = mmlu_score_from_tsv(args.results_tsv, args.variant or "")
    mmlu["score"] = score
    if score is None:
        warn("MMLU score is null (pass --mmlu-score, --eval-dir, or --results-tsv)")

    # ---- environment variables used in test ----
    env_names = list(dict.fromkeys(DEFAULT_ENV_VARS + (args.env or [])))
    env_forced = _parse_kv_list(args.env_set)
    env_vars = {}
    for name in dict.fromkeys(list(env_forced) + env_names):
        if name in env_forced:
            env_vars[name] = env_forced[name]
        else:
            env_vars[name] = os.environ.get(name)

    return {
        "model": args.variant or os.path.basename(args.bench_json),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_versions": versions,
        "olive_recipe": recipe,
        "provider_options": provider,
        "mmlu": mmlu,
        "environment_variables": env_vars,
        "benchmark": backbone,
    }


def derive_defaults(args):
    """Fill path defaults from --variant relative to the script dir."""
    v = args.variant
    if v:
        args.bench_json = args.bench_json or os.path.join(SCRIPT_DIR, f"bench_{v}.json")
        args.model_dir = args.model_dir or os.path.join(SCRIPT_DIR, "variants", v)
        args.recipe = args.recipe or os.path.join(SCRIPT_DIR, f"gpt-oss-20b_{v}.json")
        args.output = args.output or os.path.join(SCRIPT_DIR, f"summary_{v}.json")
    if not args.bench_json:
        raise SystemExit("error: --bench-json is required (or pass --variant to derive it)")
    if not args.model_dir and args.bench_json:
        # last resort: try variants/<name-from-bench>
        base = os.path.basename(args.bench_json)
        if base.startswith("bench_") and base.endswith(".json"):
            name = base[len("bench_"):-len(".json")]
            args.variant = args.variant or name
            args.model_dir = os.path.join(SCRIPT_DIR, "variants", name)
            args.recipe = args.recipe or os.path.join(SCRIPT_DIR, f"gpt-oss-20b_{name}.json")
    args.genai_config = args.genai_config or (
        os.path.join(args.model_dir, "genai_config.json") if args.model_dir else None
    )
    if not args.output:
        stem = os.path.splitext(os.path.basename(args.bench_json))[0]
        args.output = os.path.join(SCRIPT_DIR, f"summary_{stem}.json")
    return args


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", help="Variant name; used to derive default paths.")
    p.add_argument("--bench-json", help="Benchmark results JSON (backbone).")
    p.add_argument("--model-dir", help="Built model dir containing genai_config.json.")
    p.add_argument("--genai-config", help="Explicit genai_config.json path.")
    p.add_argument("--recipe", help="Olive recipe JSON used to build the model.")
    p.add_argument("--output", help="Output summary JSON path.")

    p.add_argument("--ort-dir", default=os.path.join(HOME, "onnxruntime"),
                   help="onnxruntime repo dir.")
    p.add_argument("--genai-dir", default=os.path.join(HOME, "onnxruntime-genai"),
                   help="onnxruntime-genai repo dir.")
    p.add_argument("--evals-dir", default=os.path.join(HOME, "evals"),
                   help="evals repo dir.")

    p.add_argument("--completion-fn", default="oss/gpt-oss-20b",
                   help="evals completion fn whose args hold the MMLU gen options.")
    p.add_argument("--mmlu-eval", default="match_mmlu", help="MMLU eval name.")
    p.add_argument("--mmlu-max-samples", type=int, default=0,
                   help="0 = full 14042-sample set.")
    p.add_argument("--mmlu-option", action="append", default=[],
                   metavar="KEY=VAL",
                   help="Override/add an MMLU generation option (repeatable).")
    p.add_argument("--mmlu-score", type=float, default=None,
                   help="Explicit MMLU accuracy (0-1).")
    p.add_argument("--eval-dir", help="Eval run dir to pool match_mmlu shards from.")
    p.add_argument("--results-tsv",
                   help="experiment_results_v2.tsv to read the MMLU score from.")

    p.add_argument("--env", action="append", default=[], metavar="NAME",
                   help="Extra env var name to snapshot from the environment.")
    p.add_argument("--env-set", action="append", default=[], metavar="NAME=VAL",
                   help="Force-record an env var value used in the test (repeatable).")

    args = p.parse_args()
    args = derive_defaults(args)

    summary = build_summary(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"[summary] wrote {args.output}")


if __name__ == "__main__":
    main()
