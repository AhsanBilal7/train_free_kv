#!/usr/bin/env python
"""Collect the ``result.json`` of each run in ``experiments/<benchmark>/pred/`` into one table.

``eval.py`` writes a ``result.json`` in each run folder. LongBench uses
``{task: score}``. RULER uses ``{length: {task: score, "average": s}, "average": s}``.
The script:

1. gets the hyperparameters of each folder with :func:`run_naming.parse_run_name`;
2. gives each folder the name of the ``config/*.conf`` whose settings agree, or a
   label made from the folder name;
3. keeps the folders that ``--config`` (union) and ``--filter`` (intersection)
   select;
4. prints the scores, writes one CSV row for each run (with a MEAN column) and,
   with ``--plot``, draws accuracy against budget with one line for each method.

Examples::

    python scripts/collect_results.py --benchmark LongBench
    python scripts/collect_results.py --benchmark RULER --config lorekv-ent-unif-topk --plot
    python scripts/collect_results.py --filter mode=causal_lookahead --filter budget=128
"""
import argparse
import json
import math
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_DIR = os.path.join(REPO_ROOT, "config")


# stable column order for the hyperparameter columns in the wide technique table.
# cell/brule/sel come first: they are the readable form of b/g/lD/tau/lK/lV.
HYPERPARAM_ORDER = ["mode", "model", "budget", "cell", "brule", "sel",
                    "chunk", "M", "T", "lL", "lT",
                    "b", "g", "lD", "nrm", "alpha", "kappa", "lK", "lV", "tau", "obj"]

# config-file variable  ->  folder-name key
CONF_TO_KEY = {
    "MODE": "mode",
    "MODEL_PATH": "model",
    "BUDGET": "budget",
    "NUM_PARTICLES": "M",
    "LOOKAHEAD_LEN": "T",
    "LAMBDA_LOOKAHEAD": "lL",
    "LOOKAHEAD_TEMPERATURE": "lT",
    "BETA": "b",
    "GAMMA": "g",
    "LAMBDA_D": "lD",
    "NORMALIZE_TYPICALITY": "nrm",
    "ROBUST_ALPHA": "alpha",
    "ROBUST_TOPK_BETA": "kappa",
    "LAMBDA_K": "lK",
    "LAMBDA_V": "lV",
    "BUDGET_TAU": "tau",
    "SCORE_OBJECTIVE": "obj",
}

sys.path.insert(0, SCRIPT_DIR)
from run_naming import parse_run_name  # noqa: E402


def norm(v):
    """Normalize a value for comparison. Numbers compare as numbers ("1" == "1.0")."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return str(v)


def parse_folder(name):
    """Return the hyperparameters of a folder name, or None."""
    return parse_run_name(name)


def load_conf(path):
    """Read a ``KEY = VALUE`` config into ``{KEY: [values]}``."""
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
    """Convert a config to a filter ``{folder_key: set of allowed values}``.

    The filter uses only the keys that the folder names of the mode contain. A
    LORE-KV setting thus does not exclude a baseline folder.
    """
    mode = conf.get("MODE", ["andpro"])[0]
    # MODEL_PATH is included so a Mistral folder does not also match the -llama
    # config: the two differ ONLY in the model, so without it every folder matched
    # both and the technique label came out as "lorekv-...-llama|lorekv-...".
    if mode == "causal_lookahead":
        allowed_conf_keys = {"MODE", "MODEL_PATH", "BUDGET", "NUM_PARTICLES", "LOOKAHEAD_LEN",
                             "LAMBDA_LOOKAHEAD",
                             "LOOKAHEAD_TEMPERATURE", "BETA", "GAMMA", "LAMBDA_D",
                             "NORMALIZE_TYPICALITY", "ROBUST_ALPHA",
                             "ROBUST_TOPK_BETA", "LAMBDA_K", "LAMBDA_V", "BUDGET_TAU",
                             "SCORE_OBJECTIVE"}
    else:
        allowed_conf_keys = {"MODE", "MODEL_PATH", "BUDGET"}

    filt = {}
    for conf_key, values in conf.items():
        if conf_key not in allowed_conf_keys:
            continue
        # A config stores the full HF id; folders carry only the basename (MODEL_TAG).
        if conf_key == "MODEL_PATH":
            values = [os.path.basename(str(v).rstrip("/")) for v in values]
        filt[CONF_TO_KEY[conf_key]] = {norm(v) for v in values}
    # App. C (iii): a causal_lookahead config that does not mention SCORE_OBJECTIVE ran
    # the default projected score, so pin it explicitly. Without this every core config
    # would also match the _obj_attention / _obj_unprojected folders and the technique
    # label for those runs would come back as "core|ablation".
    if mode == "causal_lookahead":
        filt.setdefault("obj", {"projected"})
    return filt


def load_named_configs(config_dir):
    """Return ``[(name, filter)]`` for each ``config/<name>.conf``."""
    configs = []
    if not os.path.isdir(config_dir):
        return configs
    for fn in sorted(os.listdir(config_dir)):
        if not fn.endswith(".conf"):
            continue
        name = fn[: -len(".conf")]
        filt = conf_to_filter(load_conf(os.path.join(config_dir, fn)))
        if filt:
            configs.append((name, filt))
    return configs


def budget_free_label(name):
    """Return a label for a folder without the budget, so that a budget sweep gives one line.

    For a LORE-KV run the label is ``lorekv-<cell>-<brule>-<sel>_<model>`` and the
    values of M, T and nrm. If the name cannot be read, the function returns the name.
    """
    p = parse_run_name(name)
    if not p:
        return name
    if p.get("mode") == "causal_lookahead" and p.get("cell"):
        rest = "_".join(f"{k}{p[k]}" for k in ("M", "T", "nrm") if p.get(k) is not None)
        if p.get("obj") not in (None, "projected"):
            rest = f"{rest}_obj{p['obj']}" if rest else f"obj{p['obj']}"
        model = p.get("model", "")
        tail = f"_{rest}" if rest else ""
        return f"lorekv-{p['cell']}-{p['brule']}-{p['sel']}_{model}{tail}"
    return f"{p.get('mode')}_{p.get('model')}_chunk_{p.get('chunk')}"


def technique_name(params, folder, named_configs):
    """Return the names of the configs that agree with a folder, joined with '|'.

    A folder agrees with a config if each value that the config sets is in the
    folder. If no config agrees, the function returns :func:`budget_free_label`.
    """
    hits = [name for name, filt in named_configs if matches(params, filt)]
    return "|".join(hits) if hits else budget_free_label(folder)


def matches(params, filt):
    """Return True if each key of ``filt`` is in ``params`` with an allowed value."""
    for key, allowed in filt.items():
        if key not in params or norm(params[key]) not in allowed:
            return False
    return True


def flatten(obj, prefix=""):
    """Convert a nested ``result.json`` to ``{path: score}``. Values that are not numbers are ignored."""
    out = {}
    for k, v in obj.items():
        key = k if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def is_average_col(name):
    """Return True for a column that is an average."""
    return name == "average" or name.endswith("/average") or name.endswith("/mean")


def order_datasets(datasets):
    """Return the dataset columns in sorted order, followed by the average columns."""
    real = sorted(d for d in datasets if not is_average_col(d))
    avg = sorted(d for d in datasets if is_average_col(d))
    return real + avg


def make_comparison_png(df, real_cols, out_path):
    """Draw one PNG with one panel for each dataset and MEAN: accuracy against budget.

    Each method is one line. Runs of the same method at the same budget are averaged.
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
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="LongBench", choices=["LongBench", "RULER", "NIAH"])
    p.add_argument("--pred-dir", default=None,
                   help="folder of run directories (default: experiments/<benchmark>/pred)")
    p.add_argument("--config", action="append", default=[], metavar="NAME",
                   help="named config to select (config/<NAME>.conf, or a path). Repeatable -> union.")
    p.add_argument("--filter", action="append", default=[], metavar="KEY=VAL",
                   help="extra hyperparameter filter, e.g. budget=128 or M=4. Repeatable -> AND.")
    p.add_argument("--out", default="results_summary.csv", help="output CSV path (default: results_summary.csv)")
    p.add_argument("--long", action="store_true",
                   help="write the long form (technique,folder,dataset,accuracy) instead of the wide technique table")
    p.add_argument("--plot", action="store_true",
                   help="also render a single PNG of per-dataset technique-vs-budget comparison graphs")
    p.add_argument("--plot-out", default="results_comparison.png",
                   help="output PNG path for --plot (default: results_comparison.png)")
    return p.parse_args()


def main():
    """Collect, filter, print and write the results."""
    args = parse_args()
    args.pred_dir = args.pred_dir or os.path.join(REPO_ROOT, "experiments", args.benchmark, "pred")
    if not os.path.isdir(args.pred_dir):
        raise SystemExit(f"No such pred dir: {args.pred_dir}")

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
