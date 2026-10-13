#!/usr/bin/env python3
"""⏱️ Average prompt length and per-sample wall-clock time over finished runs (Table 3).

For every (benchmark, config, budget) this locates the run folder that
``scripts/sweep.py`` writes for that config (names come from ``run_naming.py``),
reads each per-dataset ``*.jsonl`` and averages the ``length`` and ``time`` fields
that every ``pred.py`` stores per sample. ``time`` covers prefill, the LORE-KV
lookahead and scoring, and decoding, i.e. the per-sample latency of Table 3.

Usage::

    python scripts/report_length_time.py                              # defaults below
    python scripts/report_length_time.py --benchmarks NIAH RULER \
        --configs andpro critical --budgets 128 1024
    python scripts/report_length_time.py --include-partial            # in-flight runs

Writes ``reports/length_time_per_dataset.csv`` and ``reports/length_time_summary.csv``.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import OrderedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIGS = [
    "andpro",
    "andpro-llama",
    "critical",
    "critical-llama",
    "lorekv-ent-unif-topk",
    "lorekv-ent-unif-topk-llama",
]
DEFAULT_BENCHMARKS = ["LongBench", "RULER", "NIAH"]
DEFAULT_BUDGETS = ["128", "256", "512", "1024"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_naming import build_run_name  # noqa: E402
from sweep import LOREKV, SHARED, load_conf  # noqa: E402

DEFAULTS = {var: default for var, _, default in SHARED + LOREKV}
DEFAULTS["MODEL_PATH"] = "mistralai/Mistral-7B-Instruct-v0.2"


def conf_get(conf, key):
    return conf.get(key) or DEFAULTS[key]


def first(value):
    """A sweep list ('128 256 512') collapses to its first element."""
    parts = str(value).split()
    return parts[0] if parts else ""


def out_names(conf, budget, niah_multi=False, legacy=True):
    """Candidate run-folder names for one config and budget, most preferred first.

    Baseline folders written by older versions of the repository carry a
    ``_red_0.0`` suffix; that spelling is offered as a fallback unless ``legacy``
    is False.
    """
    g = lambda k: first(conf.get(k) or DEFAULTS[k])  # noqa: E731
    mode = g("MODE")
    model_tag = os.path.basename(g("MODEL_PATH").rstrip("/"))
    name = build_run_name(
        mode=mode, model_tag=model_tag, budget=budget, chunk_size=g("CHUNK_SIZE"),
        num_particles=g("NUM_PARTICLES"), lookahead_len=g("LOOKAHEAD_LEN"),
        lambda_lookahead=g("LAMBDA_LOOKAHEAD"),
        lookahead_temperature=g("LOOKAHEAD_TEMPERATURE"),
        beta=g("BETA"), gamma=g("GAMMA"), lambda_d=g("LAMBDA_D"),
        normalize_typicality=g("NORMALIZE_TYPICALITY"),
        robust_alpha=g("ROBUST_ALPHA"), robust_topk_beta=g("ROBUST_TOPK_BETA"),
        lambda_k=g("LAMBDA_K"), lambda_v=g("LAMBDA_V"), budget_tau=g("BUDGET_TAU"),
        score_objective=g("SCORE_OBJECTIVE"), name_prefix=conf.get("RUN_NAME_PREFIX") or None)
    names = [name]
    if mode != "causal_lookahead" and legacy:
        names.append(f"{name}_red_0.0")
    return [f"multi_{n}" for n in names] if niah_multi else names


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a *_tmp checkpoint can end mid-write if the job was killed
                continue
    return rows


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    return statistics.fmean(values) if values else None


def summarize_file(path, status):
    """Per-dataset stats for one <dataset>.jsonl."""
    rows = read_jsonl(path)
    if not rows:
        return None

    lengths = [v for v in (as_float(r.get("length")) for r in rows) if v is not None]
    times = [v for v in (as_float(r.get("time")) for r in rows) if v is not None]
    # NIAH stores the tokens actually fed to the model; RULER's *_tmp rows do too.
    inputs = [
        v
        for v in (
            as_float(r.get("input_length", r.get("context_length"))) for r in rows
        )
        if v is not None
    ]

    return {
        "num_samples": len(rows),
        "avg_length": mean(lengths),
        "median_length": statistics.median(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "avg_input_tokens": mean(inputs),
        "avg_time_s": mean(times),
        "median_time_s": statistics.median(times) if times else None,
        "max_time_s": max(times) if times else None,
        "total_time_s": sum(times) if times else None,
        "status": status,
        # kept for the weighted roll-up, dropped before writing
        "_lengths": lengths,
        "_times": times,
        "_inputs": inputs,
    }


def dataset_files(run_dir, include_partial):
    """Map dataset name -> (file path, status), preferring finished .jsonl."""
    found = OrderedDict()
    for entry in sorted(os.listdir(run_dir)):
        if entry.endswith(".jsonl"):
            found[entry[: -len(".jsonl")]] = (os.path.join(run_dir, entry), "complete")
    if include_partial:
        for entry in sorted(os.listdir(run_dir)):
            if entry.endswith(".jsonl_tmp"):
                name = entry[: -len(".jsonl_tmp")]
                if name not in found:
                    found[name] = (os.path.join(run_dir, entry), "partial")
    return found


def fmt(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(
        description="CSV report of average sample length and generation time per "
        "prediction run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo-root", default=REPO_ROOT)
    ap.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="config/<name>.conf stems (default: the six paper configs)",
    )
    ap.add_argument(
        "--budgets",
        nargs="+",
        default=DEFAULT_BUDGETS,
        help="KV budgets to look for; 'from-config' uses each config's BUDGET line",
    )
    ap.add_argument(
        "--include-partial",
        action="store_true",
        help="also read *.jsonl_tmp checkpoints of unfinished runs",
    )
    ap.add_argument(
        "--no-legacy-names",
        action="store_true",
        help="do not fall back to pre-'_red_<x>' folder names for andpro/critical runs",
    )
    ap.add_argument(
        "--niah-task",
        choices=["single", "multi", "both"],
        default="single",
        help="NIAH runs are written to multi_<out_name> when NIAH_TASK=multi",
    )
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "reports"))
    ap.add_argument("--per-dataset-csv", default="length_time_per_dataset.csv")
    ap.add_argument("--summary-csv", default="length_time_summary.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_dataset_rows = []
    summary_rows = []
    missing = []

    for cfg_name in args.configs:
        cfg_path = os.path.join(args.repo_root, "config", f"{cfg_name}.conf")
        if not os.path.isfile(cfg_path):
            print(f"[warn] no such config: {cfg_path}", file=sys.stderr)
            continue
        conf = load_conf(cfg_path)
        mode = first(conf_get(conf, "MODE"))
        model_tag = os.path.basename(conf_get(conf, "MODEL_PATH").rstrip("/"))
        budgets = (
            conf_get(conf, "BUDGET").split()
            if args.budgets == ["from-config"]
            else args.budgets
        )

        for benchmark in args.benchmarks:
            pred_root = os.path.join(args.repo_root, "experiments", benchmark, "pred")
            if not os.path.isdir(pred_root):
                print(f"[warn] no pred/ dir: {pred_root}", file=sys.stderr)
                continue

            if benchmark == "NIAH":
                variants = {"single": [False], "multi": [True], "both": [False, True]}[
                    args.niah_task
                ]
            else:
                variants = [False]

            for budget in budgets:
                for is_multi in variants:
                    candidates = out_names(
                        conf, budget, niah_multi=is_multi, legacy=not args.no_legacy_names
                    )
                    task = "multi" if is_multi else "single"

                    run_dir, files = None, {}
                    for name in candidates:
                        candidate_dir = os.path.join(pred_root, name)
                        if not os.path.isdir(candidate_dir):
                            continue
                        candidate_files = dataset_files(candidate_dir, args.include_partial)
                        if candidate_files:
                            run_dir, files = candidate_dir, candidate_files
                            break
                        run_dir = run_dir or candidate_dir  # exists but empty
                    if run_dir is None:
                        missing.append(
                            f"{benchmark}/{cfg_name}/b{budget} -> {candidates[0]}"
                        )
                        continue

                    stats_by_dataset = {}
                    for dataset, (path, status) in files.items():
                        stats = summarize_file(path, status)
                        if stats is None:
                            continue
                        stats_by_dataset[dataset] = stats
                        row = OrderedDict(
                            benchmark=benchmark,
                            config=cfg_name,
                            mode=mode,
                            model=model_tag,
                            budget=budget,
                            task=task if benchmark == "NIAH" else "",
                            dataset=dataset,
                            num_samples=stats["num_samples"],
                            avg_length=fmt(stats["avg_length"], 1),
                            median_length=fmt(stats["median_length"], 1),
                            max_length=fmt(stats["max_length"], 1),
                            avg_input_tokens=fmt(stats["avg_input_tokens"], 1),
                            avg_time_s=fmt(stats["avg_time_s"], 4),
                            median_time_s=fmt(stats["median_time_s"], 4),
                            max_time_s=fmt(stats["max_time_s"], 4),
                            total_time_s=fmt(stats["total_time_s"], 2),
                            status=stats["status"],
                            run_dir=os.path.relpath(run_dir, args.repo_root),
                        )
                        per_dataset_rows.append(row)

                    if not stats_by_dataset:
                        missing.append(
                            f"{benchmark}/{cfg_name}/b{budget} -> "
                            f"{os.path.basename(run_dir)} (no jsonl)"
                        )
                        continue

                    # Roll-up: pooled over every sample, not a mean of per-dataset means.
                    all_lengths = [
                        v for s in stats_by_dataset.values() for v in s["_lengths"]
                    ]
                    all_times = [
                        v for s in stats_by_dataset.values() for v in s["_times"]
                    ]
                    all_inputs = [
                        v for s in stats_by_dataset.values() for v in s["_inputs"]
                    ]
                    statuses = {s["status"] for s in stats_by_dataset.values()}
                    summary_rows.append(
                        OrderedDict(
                            benchmark=benchmark,
                            config=cfg_name,
                            mode=mode,
                            model=model_tag,
                            budget=budget,
                            task=task if benchmark == "NIAH" else "",
                            num_datasets=len(stats_by_dataset),
                            num_samples=sum(
                                s["num_samples"] for s in stats_by_dataset.values()
                            ),
                            avg_length=fmt(mean(all_lengths), 1),
                            avg_input_tokens=fmt(mean(all_inputs), 1),
                            avg_time_s=fmt(mean(all_times), 4),
                            median_time_s=fmt(
                                statistics.median(all_times) if all_times else None, 4
                            ),
                            max_time_s=fmt(max(all_times) if all_times else None, 4),
                            total_time_s=fmt(sum(all_times) if all_times else None, 2),
                            total_time_min=fmt(
                                sum(all_times) / 60.0 if all_times else None, 2
                            ),
                            status="complete"
                            if statuses == {"complete"}
                            else "+".join(sorted(statuses)),
                            datasets=";".join(stats_by_dataset),
                            run_dir=os.path.relpath(run_dir, args.repo_root),
                        )
                    )

    def write(path, rows):
        if not rows:
            print(f"[warn] nothing to write to {path}", file=sys.stderr)
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows):4d} rows -> {path}")

    write(os.path.join(args.out_dir, args.per_dataset_csv), per_dataset_rows)
    write(os.path.join(args.out_dir, args.summary_csv), summary_rows)

    if missing:
        print(f"\n{len(missing)} run folder(s) not found / empty:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)

    if summary_rows:
        head = ("benchmark", "config", "budget", "num_samples", "avg_length",
                "avg_time_s", "total_time_min", "status")
        widths = [12, 28, 7, 8, 11, 11, 15, 18]
        print()
        print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
        for row in summary_rows:
            print("  ".join(str(row[h]).ljust(w) for h, w in zip(head, widths)))


if __name__ == "__main__":
    main()
