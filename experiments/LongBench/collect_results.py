#!/usr/bin/env python
"""Collect accuracy scores from every pred/<config>/result.json into one table.

Works unchanged inside experiments/LongBench/ and experiments/RULER/ (each has its
own pred/ folder). Each sub-folder of pred/ is named by the run's hyperparameters
(the `out_name` scheme from run.sh) and holds a result.json produced by eval.py:

  * LongBench -> flat   {"<task>": score, ...}
  * RULER     -> nested {"<length>": {"<task>": score, ..., "average": s}, ...,
                         "average": overall}

This script:
  1. Parses each folder name back into its hyperparameters
     (mode / budget / chunk / M / T / lL / lT / lH / eta / alpha / beta / lK / lV / tau,
      or red for the non-lookahead modes).
  2. Optionally keeps only the folders matching a hyperparameter selection -- given
     either as a named config you run experiments with (`--config tlckv-pe-m4-adap-topk`,
     read straight from config/<name>.conf) and/or ad-hoc `--filter key=value` pairs.
  3. Prints the accuracies per folder to the terminal.
  4. Writes them to a CSV (long form by default: columns folder,dataset,accuracy;
     `--wide` pivots to folder x dataset instead).

Examples
--------
  # everything under pred/
  python collect_results.py

  # only the folders produced by a specific named config (across its budget sweep)
  python collect_results.py --config tlckv-pe-m4-adap-topk

  # ad-hoc hyperparameter selection
  python collect_results.py --filter mode=causal_lookahead --filter budget=128 --filter lH=0.3

  # scope a config to a single budget and write a wide (folder x dataset) csv
  python collect_results.py --config tlckv-pe-m4-adap-topk --filter budget=256 --wide
"""
import argparse
import json
import os
import re

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "config"))

# folder-name token -> nothing (the token IS the key); listed for reference/order
LOOKAHEAD_KEYS = ["chunk", "M", "T", "lL", "lT", "lH", "eta", "alpha", "beta", "lK", "lV", "tau"]

# config-file variable  ->  folder-name key
CONF_TO_KEY = {
    "MODE": "mode",
    "BUDGET": "budget",
    "NUM_PARTICLES": "M",
    "LOOKAHEAD_LEN": "T",
    "LAMBDA_LOOKAHEAD": "lL",
    "LOOKAHEAD_TEMPERATURE": "lT",
    "LAMBDA_H": "lH",
    "PARTICLE_ETA": "eta",
    "ROBUST_ALPHA": "alpha",
    "ROBUST_TOPK_BETA": "beta",
    "LAMBDA_K": "lK",
    "LAMBDA_V": "lV",
    "BUDGET_TAU": "tau",
    "LAMBDA_RED": "red",
}

# folder name: <mode>_<model>_<budget>_chunk_<cs>[<tail>]
_NAME_RE = re.compile(r"^(?P<mode>.+)_(?P<model>[^_]+)_(?P<budget>\d+)_chunk_(?P<chunk>[^_]+)(?P<tail>.*)$")


def norm(v):
    """Normalise a value for comparison: numbers compare numerically ("1" == "1.0")."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return str(v)


def parse_folder(name):
    """Folder name -> {mode, model, budget, chunk, ...}. Returns None if unparseable."""
    m = _NAME_RE.match(name)
    if not m:
        return None
    params = {"mode": m.group("mode"), "model": m.group("model"),
              "budget": m.group("budget"), "chunk": m.group("chunk")}
    tail = m.group("tail")
    if tail:
        toks = tail.strip("_").split("_")
        for i in range(0, len(toks) - 1, 2):
            params[toks[i]] = toks[i + 1]
    return params


def load_conf(path):
    """Parse a KEY = VALUE .conf file -> {KEY: [values...]} (space-separated lists)."""
    conf = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            conf[key.strip()] = val.strip().split()
    return conf


def conf_to_filter(conf):
    """A parsed .conf -> {folder_key: set(normalised allowed values)}.

    Uses only the keys that actually appear in the corresponding folder name for
    that mode (lookahead scheme vs. the plain <...>_red_<x> scheme), so e.g. a
    LAMBDA_RED line in a causal_lookahead config doesn't wrongly exclude every
    lookahead folder (which has no `red` token).
    """
    mode = conf.get("MODE", ["andpro"])[0]
    if mode == "causal_lookahead":
        allowed_conf_keys = {"MODE", "BUDGET", "NUM_PARTICLES", "LOOKAHEAD_LEN", "LAMBDA_LOOKAHEAD",
                             "LOOKAHEAD_TEMPERATURE", "LAMBDA_H", "PARTICLE_ETA", "ROBUST_ALPHA",
                             "ROBUST_TOPK_BETA", "LAMBDA_K", "LAMBDA_V", "BUDGET_TAU"}
    else:
        allowed_conf_keys = {"MODE", "BUDGET", "LAMBDA_RED"}

    filt = {}
    for conf_key, values in conf.items():
        if conf_key not in allowed_conf_keys:
            continue
        filt[CONF_TO_KEY[conf_key]] = {norm(v) for v in values}
    return filt


def matches(params, filt):
    """True iff every key in `filt` is present in params with an allowed value."""
    for key, allowed in filt.items():
        if key not in params or norm(params[key]) not in allowed:
            return False
    return True


def flatten(obj, prefix=""):
    """Nested result.json -> {dataset_path: numeric_score}. Non-numeric leaves skipped."""
    out = {}
    for k, v in obj.items():
        key = k if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def is_average_col(name):
    return name == "average" or name.endswith("/average") or name.endswith("/mean")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", default="pred", help="directory holding the per-config result folders (default: pred)")
    p.add_argument("--config", action="append", default=[], metavar="NAME",
                   help="named config to select (config/<NAME>.conf, or a path). Repeatable -> union.")
    p.add_argument("--filter", action="append", default=[], metavar="KEY=VAL",
                   help="extra hyperparameter filter, e.g. budget=128 or lH=0.3. Repeatable -> AND.")
    p.add_argument("--out", default="results_summary.csv", help="output CSV path (default: results_summary.csv)")
    p.add_argument("--wide", action="store_true", help="write a wide folder x dataset CSV instead of long form")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.pred_dir):
        raise SystemExit(f"No such pred dir: {args.pred_dir} (run from an experiment folder, e.g. experiments/RULER)")

    # ---- build the hyperparameter selection ----------------------------------
    config_filters = []
    for name in args.config:
        path = name if (name.endswith(".conf") or os.sep in name) else os.path.join(CONFIG_DIR, f"{name}.conf")
        if not os.path.isfile(path):
            raise SystemExit(f"Config not found: {path}")
        config_filters.append((name, conf_to_filter(load_conf(path))))

    global_filter = {}
    for item in args.filter:
        if "=" not in item:
            raise SystemExit(f"Bad --filter (expected KEY=VAL): {item}")
        k, _, v = item.partition("=")
        global_filter[k.strip()] = {norm(v.strip())}

    if config_filters:
        print("Selecting configs:", ", ".join(n for n, _ in config_filters))
    if global_filter:
        print("Extra filters:", ", ".join(f"{k}={sorted(v)}" for k, v in global_filter.items()))

    # ---- scan pred/ ----------------------------------------------------------
    records = []          # (folder, dataset, accuracy)
    kept, skipped = [], []
    for folder in sorted(os.listdir(args.pred_dir)):
        fdir = os.path.join(args.pred_dir, folder)
        result_path = os.path.join(fdir, "result.json")
        if not os.path.isdir(fdir) or not os.path.isfile(result_path):
            continue

        params = parse_folder(folder)
        if params is None:
            skipped.append((folder, "unparseable name"))
            continue

        # config filters are OR-ed (match ANY selected config); global filters AND-ed on top
        if config_filters and not any(matches(params, f) for _, f in config_filters):
            continue
        if not matches(params, global_filter):
            continue

        with open(result_path) as f:
            scores = flatten(json.load(f))
        if not scores:
            skipped.append((folder, "no numeric scores"))
            continue

        kept.append(folder)
        for dataset, acc in scores.items():
            records.append({"folder": folder, "dataset": dataset, "accuracy": round(acc, 2)})

    if not records:
        print("\nNo matching result.json found for the given selection.")
        if skipped:
            print("Skipped:", "; ".join(f"{f} ({why})" for f, why in skipped))
        return

    # ---- terminal report -----------------------------------------------------
    df = pd.DataFrame(records)
    print(f"\nMatched {len(kept)} config folder(s) under {args.pred_dir}/\n")
    for folder in kept:
        sub = df[df.folder == folder]
        print(f"=== {folder} ===")
        width = max(len(d) for d in sub.dataset)
        for _, row in sub.iterrows():
            print(f"    {row.dataset:<{width}} : {row.accuracy:>7.2f}")
        real = sub[~sub.dataset.map(is_average_col)]
        if not real.empty:
            print(f"    {'MEAN (computed)':<{width}} : {real.accuracy.mean():>7.2f}")
        print()

    # ---- csv -----------------------------------------------------------------
    if args.wide:
        table = df.pivot_table(index="folder", columns="dataset", values="accuracy")
        table.to_csv(args.out)
    else:
        df.to_csv(args.out, index=False)
    print(f"Wrote {'wide' if args.wide else 'long'} CSV -> {args.out}  ({len(df)} rows)")
    if skipped:
        print("Skipped folders:", "; ".join(f"{f} ({why})" for f, why in skipped))


if __name__ == "__main__":
    main()
