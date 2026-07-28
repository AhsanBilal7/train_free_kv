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
  2. Resolves a *technique name* for each folder by matching its hyperparameters
     against every config/<name>.conf: the folder is labelled with the name of the
     config(s) whose hyperparameter combination it belongs to (e.g. the ablation
     codenames A1/A2/B1/... under config/tlckv-*.conf). Folders that match no
     named config fall back to their folder name as the technique.
  3. Optionally keeps only the folders matching a hyperparameter selection -- given
     either as a named config you run experiments with (`--config tlckv-pe-m4-adap-topk`,
     read straight from config/<name>.conf) and/or ad-hoc `--filter key=value` pairs.
  4. Prints the accuracies per technique to the terminal.
  5. Writes ONE row per run to a CSV: the technique name, its hyperparameter
     columns, then one accuracy column per dataset (plus a computed MEAN).
     `--long` instead writes the old long form (columns technique,folder,dataset,accuracy).
  6. With `--plot`, also renders a SINGLE PNG holding a grid of comparison graphs
     -- one panel per dataset (plus MEAN), each plotting accuracy vs. KV budget
     with one line per technique -- so techniques can be compared across every
     dataset and every budget at a glance.

Examples
--------
  # everything under pred/  -> technique x (hyperparameters + datasets) table
  python collect_results.py

  # also drop a single comparison PNG (technique lines, per-dataset panels, x=budget)
  python collect_results.py --plot

  # only the folders produced by a specific named config (across its budget sweep)
  python collect_results.py --config tlckv-pe-m4-adap-topk

  # ad-hoc hyperparameter selection
  python collect_results.py --filter mode=causal_lookahead --filter budget=128 --filter lH=0.3

  # scope a config to a single budget and write the long form instead
  python collect_results.py --config tlckv-pe-m4-adap-topk --filter budget=256 --long
"""
import argparse
import json
import math
import os
import re

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "config"))

# folder-name token -> nothing (the token IS the key); listed for reference/order
LOOKAHEAD_KEYS = ["chunk", "M", "T", "lL", "lT", "lH", "eta", "alpha", "beta", "lK", "lV", "tau"]

# stable column order for the hyperparameter columns in the wide technique table
HYPERPARAM_ORDER = ["mode", "model", "budget", "chunk", "M", "T", "lL", "lT", "lH",
                    "eta", "alpha", "beta", "lK", "lV", "tau", "red"]

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


def load_named_configs(config_dir):
    """Every config/<name>.conf (except default.conf) -> [(name, filter_dict), ...].

    These are the named techniques (ablation codenames A1/A2/B1/... etc.) a folder
    can be labelled with. default.conf is skipped: it is the generic template, not a
    distinct technique, so it would otherwise duplicate whichever config it mirrors.
    """
    configs = []
    if not os.path.isdir(config_dir):
        return configs
    for fn in sorted(os.listdir(config_dir)):
        if not fn.endswith(".conf") or fn == "default.conf":
            continue
        name = fn[: -len(".conf")]
        filt = conf_to_filter(load_conf(os.path.join(config_dir, fn)))
        if filt:
            configs.append((name, filt))
    return configs


def budget_free_label(name):
    """Folder name with the budget token dropped -> a technique label shared across
    a budget sweep (budget lives in its own column, so runs that differ only in
    budget collapse onto one technique / one line in the comparison plot)."""
    m = _NAME_RE.match(name)
    if not m:
        return name
    return f"{m.group('mode')}_{m.group('model')}_chunk_{m.group('chunk')}{m.group('tail')}"


def technique_name(params, folder, named_configs):
    """Technique for a folder: the matching config name(s), else a budget-free
    label derived from the folder name.

    A folder matches a config when every hyperparameter the config pins is present
    in the folder with an allowed value. Overlapping ablations can both claim a
    folder (e.g. a M=4 run belongs to both the fixed-M=4 and the M in {2,4,8}
    sweeps); such names are joined with '|' rather than silently picking one.
    """
    hits = [name for name, filt in named_configs if matches(params, filt)]
    return "|".join(hits) if hits else budget_free_label(folder)


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


def order_datasets(datasets):
    """Real datasets first (sorted), then the average/mean roll-up columns."""
    real = sorted(d for d in datasets if not is_average_col(d))
    avg = sorted(d for d in datasets if is_average_col(d))
    return real + avg


def make_comparison_png(df, real_cols, out_path):
    """One PNG, a grid of panels (one per dataset + MEAN): accuracy vs. KV budget,
    one line per technique. Runs with the same technique repeated at a budget
    (e.g. an M-sweep) are averaged to a single point so each technique is one line.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless: no display on the login/cluster node
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Plot skipped: matplotlib is not installed (pip install matplotlib).")
        return

    panels = real_cols + (["MEAN"] if "MEAN" in df.columns else [])
    if "budget" not in df.columns or not panels:
        print("Plot skipped: need a 'budget' column and at least one dataset.")
        return

    work = df.copy()
    work["budget_num"] = work["budget"].map(norm)
    budgets = sorted(b for b in work["budget_num"].unique() if isinstance(b, float))
    techniques = sorted(work["technique"].unique())
    if not budgets:
        print("Plot skipped: no numeric budgets to place on the x-axis.")
        return

    cmap = plt.get_cmap("tab20" if len(techniques) > 10 else "tab10")
    colors = {t: cmap(i % cmap.N) for i, t in enumerate(techniques)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
    tmark = {t: markers[i % len(markers)] for i, t in enumerate(techniques)}

    ncols = min(4, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for idx, panel in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        for t in techniques:
            g = (work[work["technique"] == t]
                 .groupby("budget_num")[panel].mean().reindex(budgets))
            xs = [b for b, y in zip(budgets, g.values) if pd.notna(y)]
            ys = [y for y in g.values if pd.notna(y)]
            if xs:
                ax.plot(xs, ys, marker=tmark[t], color=colors[t], linewidth=1.6, markersize=5)
        ax.set_title(panel, fontsize=10)
        ax.set_xlabel("KV budget")
        ax.set_ylabel("accuracy")
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(int(b)) for b in budgets])
        ax.grid(True, alpha=0.3)

    for j in range(len(panels), nrows * ncols):   # blank the unused grid cells
        axes[j // ncols][j % ncols].axis("off")

    handles = [Line2D([0], [0], color=colors[t], marker=tmark[t], linewidth=1.6,
                      markersize=5, label=t) for t in techniques]
    fig.legend(handles, techniques, loc="lower center", ncol=min(3, len(techniques)),
               fontsize=8, frameon=False)
    fig.suptitle("Technique comparison across datasets and KV budgets", fontsize=13)
    legend_frac = min(0.30, 0.05 * math.ceil(len(techniques) / 3))
    fig.tight_layout(rect=[0, legend_frac, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote comparison PNG -> {out_path}  ({len(panels)} panels, {len(techniques)} techniques)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", default="pred", help="directory holding the per-config result folders (default: pred)")
    p.add_argument("--config", action="append", default=[], metavar="NAME",
                   help="named config to select (config/<NAME>.conf, or a path). Repeatable -> union.")
    p.add_argument("--filter", action="append", default=[], metavar="KEY=VAL",
                   help="extra hyperparameter filter, e.g. budget=128 or lH=0.3. Repeatable -> AND.")
    p.add_argument("--out", default="results_summary.csv", help="output CSV path (default: results_summary.csv)")
    p.add_argument("--long", action="store_true",
                   help="write the long form (technique,folder,dataset,accuracy) instead of the wide technique table")
    p.add_argument("--plot", action="store_true",
                   help="also render a single PNG of per-dataset technique-vs-budget comparison graphs")
    p.add_argument("--plot-out", default="results_comparison.png",
                   help="output PNG path for --plot (default: results_comparison.png)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.pred_dir):
        raise SystemExit(f"No such pred dir: {args.pred_dir} (run from an experiment folder, e.g. experiments/RULER)")

    # ---- named techniques (for labelling every folder) -----------------------
    named_configs = load_named_configs(CONFIG_DIR)

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
    rows = []             # one dict per kept folder: {technique, folder, <params>, <dataset>: acc}
    all_datasets = set()
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
        all_datasets.update(scores)
        row = {"technique": technique_name(params, folder, named_configs), "folder": folder}
        row.update(params)
        row.update({d: round(a, 2) for d, a in scores.items()})
        rows.append(row)

    if not rows:
        print("\nNo matching result.json found for the given selection.")
        if skipped:
            print("Skipped:", "; ".join(f"{f} ({why})" for f, why in skipped))
        return

    # column layout: technique | present hyperparameters | datasets | MEAN
    hyper_cols = [c for c in HYPERPARAM_ORDER if any(c in r for r in rows)]
    dataset_cols = order_datasets(all_datasets)
    real_cols = [d for d in dataset_cols if not is_average_col(d)]

    df = pd.DataFrame(rows)
    for d in dataset_cols:                       # ensure every dataset column exists
        if d not in df.columns:
            df[d] = float("nan")
    if real_cols:
        df["MEAN"] = df[real_cols].mean(axis=1).round(2)
    df = df.sort_values(["technique"] + [c for c in ("budget", "folder") if c in df.columns],
                        key=lambda s: s.map(norm) if s.name in ("budget",) else s)

    ordered = ["technique"] + hyper_cols + dataset_cols + (["MEAN"] if real_cols else [])
    wide = df[ordered]

    # ---- terminal report -----------------------------------------------------
    print(f"\nMatched {len(kept)} run folder(s) under {args.pred_dir}/\n")
    for _, r in df.iterrows():
        hyp = "  ".join(f"{c}={r[c]}" for c in hyper_cols if pd.notna(r.get(c)))
        print(f"=== {r['technique']} ===")
        print(f"    [{r['folder']}]")
        if hyp:
            print(f"    {hyp}")
        width = max((len(d) for d in dataset_cols), default=0)
        for d in dataset_cols:
            if pd.notna(r.get(d)):
                print(f"    {d:<{width}} : {r[d]:>7.2f}")
        if real_cols and pd.notna(r.get("MEAN")):
            print(f"    {'MEAN (computed)':<{width}} : {r['MEAN']:>7.2f}")
        print()

    # ---- csv -----------------------------------------------------------------
    if args.long:
        long_rows = []
        for _, r in df.iterrows():
            for d in dataset_cols:
                if pd.notna(r.get(d)):
                    long_rows.append({"technique": r["technique"], "folder": r["folder"],
                                      "dataset": d, "accuracy": r[d]})
        out_df = pd.DataFrame(long_rows)
        out_df.to_csv(args.out, index=False)
        print(f"Wrote long CSV -> {args.out}  ({len(out_df)} rows)")
    else:
        wide.to_csv(args.out, index=False)
        print(f"Wrote wide technique CSV -> {args.out}  ({len(wide)} rows x {len(ordered)} cols)")

    if args.plot:
        make_comparison_png(df, real_cols, args.plot_out)

    if skipped:
        print("Skipped folders:", "; ".join(f"{f} ({why})" for f, why in skipped))


if __name__ == "__main__":
    main()
