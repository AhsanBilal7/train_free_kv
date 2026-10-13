#!/usr/bin/env python3
"""Run one benchmark for each combination of the sweep values in a config.

A config file (``config/<name>.conf``) contains ``KEY = VALUE`` lines. A value can
list several settings separated by spaces (``BUDGET = 128 256 512 1024``). Each
combination is one run. An environment variable with the same name replaces the
value in the file::

    python scripts/sweep.py --benchmark LongBench --config config/lorekv-ent-unif-topk.conf
    BUDGET="128 512" NUM_PARTICLES="1 4" python scripts/sweep.py --benchmark RULER \\
        --config config/lorekv-ent-unif-topk.conf

For each combination the script makes the run-folder name with
:mod:`run_naming` and runs ``experiments/<benchmark>/pred.py`` in the directory of
the benchmark. ``pred.py`` skips each dataset whose output exists, so a new start
continues an interrupted sweep. After all runs, ``eval.py`` scores all folders in
``pred/``.

Order of precedence: environment variable, then config file, then default.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from run_naming import build_run_name  # noqa: E402

# (variable, pred.py flag, default). Defaults are the paper's core LORE-KV /E
# configuration (Table 4). Each value may be a space-separated sweep list.
SHARED = [
    ("BUDGET", "--budget", "128 256 512 1024"),
    ("CHUNK_SIZE", "--chunk_size", "4"),
    ("MODE", "--mode", "causal_lookahead"),
]
LOREKV = [
    ("NUM_PARTICLES", "--num_particles", "4"),
    ("LOOKAHEAD_LEN", "--lookahead_len", "8"),
    ("LAMBDA_LOOKAHEAD", "--lambda_lookahead", "1.0"),
    ("LOOKAHEAD_TEMPERATURE", "--lookahead_temperature", "1.0"),
    ("BETA", "--beta", "1.0"),
    ("GAMMA", "--gamma", "1.3"),
    ("LAMBDA_D", "--lambda_d", "0.0"),
    ("NORMALIZE_TYPICALITY", "--normalize_typicality", "0"),
    ("ROBUST_ALPHA", "--robust_alpha", "0.0"),
    ("ROBUST_TOPK_BETA", "--robust_topk_beta", "1.0"),
    ("LAMBDA_K", "--lambda_k", "0.0"),
    ("LAMBDA_V", "--lambda_v", "0.0"),
    ("BUDGET_TAU", "--budget_tau", "0.0"),
    ("SCORE_OBJECTIVE", "--score_objective", "projected"),
]

LONGBENCH_TASKS = ("qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa musique "
                   "trec triviaqa samsum gov_report qmsum multi_news "
                   "passage_count passage_retrieval_en lcc repobench-p")


def load_conf(path):
    """Read a ``KEY = VALUE`` config. A ``#`` after whitespace starts a comment."""
    conf = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (x.strip() for x in line.split("=", 1))
            for sep in (" #", "\t#"):
                value = value.split(sep, 1)[0].strip()
            if value.startswith("#"):
                value = ""
            conf[key] = value
    return conf


class Settings:
    """Get a setting from the environment, then the config file, then the default."""

    def __init__(self, conf):
        self.conf = conf

    def get(self, key, default=""):
        if key in os.environ:
            return os.environ[key]
        return self.conf.get(key, default)

    def values(self, key, default):
        return self.get(key, default).split()


def benchmark_args(benchmark, s):
    """Return ``(max_length, pred.py arguments, out-name prefix)`` for one benchmark."""
    if benchmark == "LongBench":
        # 64000 exceeds every LongBench prompt, i.e. no truncation. Qwen2.5-14B's
        # configs cap it at its 32K window (App. E.3).
        max_length = s.get("MAX_LENGTH", "64000")
        tasks = s.get("DATASET_NAME_LIST") or LONGBENCH_TASKS
        args = ["--dataset", s.get("DATASET_PATH", "THUDM/LongBench"),
                "--dataset_name_list", *tasks.split()]
        return max_length, args, ""
    if benchmark == "RULER":
        lengths = s.get("DATASET_NAME_LIST") or "4096 8192 16384"
        local = s.get("RULER_LOCAL_DIR", os.path.join(REPO_ROOT, "data", "ruler_long"))
        args = ["--dataset", s.get("DATASET_PATH", "simonjegou/ruler"),
                "--local_dataset", local, "--dataset_name_list", *lengths.split()]
        # Llama-3.1's full window: RULER needles can sit anywhere, so a prompt must
        # never be middle-truncated.
        return s.get("MAX_LENGTH", "131072"), args, ""
    if benchmark == "NIAH":
        lengths = (s.get("DATASET_NAME_LIST") or "4096 8192 16384 32768").split()
        task = s.get("NIAH_TASK", "single")
        args = ["--dataset_name_list", *lengths,
                "--haystack_dir", s.get("HAYSTACK_DIR", os.path.join(
                    REPO_ROOT, "experiments", "NIAH", "PaulGrahamEssays")),
                "--depth_percents", *s.get("NIAH_DEPTHS", "0 10 20 30 40 50 60 70 80 90 100").split(),
                "--max_gen", s.get("NIAH_MAX_GEN", "64"),
                "--task", task]
        for key, flag in (("NIAH_NEEDLE", "--needle"), ("NIAH_ANSWER", "--needle_answer"),
                          ("NIAH_QUESTION", "--question")):
            if s.get(key):
                args += [flag, s.get(key)]
        if s.get("NIAH_GUIDANCE"):
            args += ["--retrieval_guidance", s.get("NIAH_GUIDANCE")]
        if task == "multi":
            for key, flag in (("NIAH_NEEDLES", "--needles"), ("NIAH_FRAGMENTS", "--needle_fragments")):
                if s.get(key):                                  # `|`-separated sentences
                    args += [flag, *s.get(key).split("|")]
        # The prompt cap must stay above the largest target length, or the needle
        # is truncated away.
        max_length = s.get("NIAH_MAX_LENGTH") or str(max(int(x) for x in lengths) + 2048)
        return max_length, args, "multi_" if task == "multi" else ""
    raise ValueError(f"unknown benchmark {benchmark!r}")


def combinations(s):
    """Yield one ``{variable: value}`` dict for each run. The budget changes slowest."""
    shared = [s.values(k, d) for k, _, d in SHARED]
    lorekv = [s.values(k, d) for k, _, d in LOREKV]
    seen = set()
    for budget, chunk, mode in itertools.product(*shared):
        if mode != "causal_lookahead":
            key = (budget, chunk, mode)
            if key not in seen:                 # baselines ignore the LORE-KV sweep
                seen.add(key)
                yield {"BUDGET": budget, "CHUNK_SIZE": chunk, "MODE": mode}
            continue
        for combo in itertools.product(*lorekv):
            row = {"BUDGET": budget, "CHUNK_SIZE": chunk, "MODE": mode}
            row.update({k: v for (k, _, _), v in zip(LOREKV, combo)})
            yield row


def run_name(row, model_tag, prefix, pred_dir):
    """Return the run-folder name of one combination.

    For a baseline the function uses an existing folder with the older suffix
    ``_red_0.0`` if the new name does not exist.
    """
    if row["MODE"] != "causal_lookahead":
        name = build_run_name(mode=row["MODE"], model_tag=model_tag,
                              budget=row["BUDGET"], chunk_size=row["CHUNK_SIZE"])
        # Folders written before lambda_red was removed carry a `_red_0.0` suffix;
        # keep writing into such a folder so finished datasets are not re-decoded.
        legacy = f"{prefix}{name}_red_0.0"
        if not os.path.isdir(os.path.join(pred_dir, prefix + name)) and \
                os.path.isdir(os.path.join(pred_dir, legacy)):
            return legacy
        return prefix + name
    return prefix + build_run_name(
        mode="causal_lookahead", model_tag=model_tag, budget=row["BUDGET"],
        chunk_size=row["CHUNK_SIZE"], num_particles=row["NUM_PARTICLES"],
        lookahead_len=row["LOOKAHEAD_LEN"], lambda_lookahead=row["LAMBDA_LOOKAHEAD"],
        lookahead_temperature=row["LOOKAHEAD_TEMPERATURE"], beta=row["BETA"],
        gamma=row["GAMMA"], lambda_d=row["LAMBDA_D"],
        normalize_typicality=row["NORMALIZE_TYPICALITY"], robust_alpha=row["ROBUST_ALPHA"],
        robust_topk_beta=row["ROBUST_TOPK_BETA"], lambda_k=row["LAMBDA_K"],
        lambda_v=row["LAMBDA_V"], budget_tau=row["BUDGET_TAU"],
        score_objective=row["SCORE_OBJECTIVE"], name_prefix=row.get("RUN_NAME_PREFIX"))


def main():
    """Run the sweep and then score the results."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", default=os.environ.get("EXPERIMENT", "LongBench"),
                    choices=["LongBench", "RULER", "NIAH"])
    ap.add_argument("--config", default=os.environ.get("CONFIG_FILE"),
                    help="config/<name>.conf; omit to use the defaults and environment only")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--no-eval", action="store_true", help="skip eval.py at the end")
    a = ap.parse_args()

    s = Settings(load_conf(a.config) if a.config else {})
    model_path = s.get("MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
    # A relative path that exists under the repository (config/*: models/<name>)
    # is a local checkpoint; anything else is passed on as a Hugging Face Hub id.
    if not os.path.isabs(model_path) and os.path.isdir(os.path.join(REPO_ROOT, model_path)):
        model_path = os.path.join(REPO_ROOT, model_path)
    model_tag = s.get("MODEL_TAG") or os.path.basename(model_path.rstrip("/"))
    bench_dir = os.path.join(REPO_ROOT, "experiments", a.benchmark)
    pred_dir = os.path.join(bench_dir, "pred")
    max_length, bench_args, prefix = benchmark_args(a.benchmark, s)
    env = dict(os.environ)
    if s.get("GPU_ID"):
        env["CUDA_VISIBLE_DEVICES"] = s.get("GPU_ID")

    print(f"[sweep] benchmark={a.benchmark} model={model_path} config={a.config}", flush=True)
    for row in combinations(s):
        row["RUN_NAME_PREFIX"] = s.get("RUN_NAME_PREFIX") or None
        name = run_name(row, model_tag, prefix, pred_dir)
        cmd = [sys.executable, "pred.py", "-m", model_path, "--max_length", max_length,
               "--out_name", name, *bench_args]
        for var, flag, _ in SHARED + (LOREKV if row["MODE"] == "causal_lookahead" else []):
            cmd += [flag, row[var]]
        print(f"[sweep] {name}\n        {shlex.join(cmd)}", flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=bench_dir, env=env, check=True)

    if not (a.dry_run or a.no_eval):
        subprocess.run([sys.executable, "eval.py"], cwd=bench_dir, env=env, check=True)


if __name__ == "__main__":
    main()
