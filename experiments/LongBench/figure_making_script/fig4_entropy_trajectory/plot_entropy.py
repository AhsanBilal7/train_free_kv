"""Figure 4 (plot stage) - reads entropy_data.npz.

Left  : per-particle predictive entropy vs single-trajectory alignment to the true
        decode queries. A downward trend = high-entropy trajectories are the bad
        ones to score eviction from.
Right : alignment when you keep a single (random) trajectory vs when you combine M
        trajectories (uniform / log-prob / LORE-KV entropy-aware). Entropy-aware
        weighting lifts the mean above a random single draw and shrinks its spread.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = {"single_mean": "Single (mean)", "single_min": "Single (worst)",
         "single_max": "Single (best)", "uniform": "Multi uniform",
         "logprob": "Multi log-prob", "lorekv": "Multi LORE-KV"}
COLOR = {"single_mean": "#1f77b4", "single_min": "#9467bd", "single_max": "#17becf",
         "uniform": "#2ca02c", "logprob": "#ff7f0e", "lorekv": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "entropy_data")
    args.out = common.png_path_for(args.data, "fig4_entropy_trajectory", args.out)
    d = np.load(args.data, allow_pickle=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ent, align = d["entropy"], d["align"]
    ax1.scatter(ent, align, s=18, alpha=0.5, color="#1f77b4", edgecolor="none")
    if len(ent) > 2:
        m, b = np.polyfit(ent, align, 1)
        xs = np.linspace(ent.min(), ent.max(), 50)
        r = np.corrcoef(ent, align)[0, 1]
        ax1.plot(xs, m * xs + b, color="#d62728", lw=2, label=f"fit (r={r:.2f})")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("Per-trajectory predictive entropy")
    ax1.set_ylabel("Single-trajectory alignment")
    ax1.set_title("High-entropy trajectories align worse")
    ax1.grid(True, alpha=0.3)

    keys = list(d["strat_keys"])
    means = d["strat_mean"]; stds = d["strat_std"]
    x = np.arange(len(keys))
    colors = [COLOR.get(k, "#333333") for k in keys]
    ax2.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL.get(k, k) for k in keys], rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Future-query alignment")
    ax2.set_title("Single draw vs combining trajectories")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Why test-time compute: multiple entropy-weighted trajectories "
                 f"(B={int(d['budget'])}, {d['dataset']})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
