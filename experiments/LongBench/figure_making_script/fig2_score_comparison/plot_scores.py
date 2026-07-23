"""Figure 2 (plot stage) - reads score_data.npz.

Left  : sorted (rank-ordered) importance-score curves for the three scoring rules
        on a representative layer -> shows the shape of each importance signal.
Right : how much the rules DISAGREE - top-B selection overlap and Spearman rank
        correlation between each pair, averaged over layers/samples.
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
LABEL = {"attention": "Attention (SnapKV-style)",
         "single_lookahead": "Single-lookahead perturbation",
         "lorekv_multi": "LORE-KV multi-lookahead"}
COLOR = {"attention": "#7f7f7f", "single_lookahead": "#1f77b4", "lorekv_multi": "#d62728"}


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "score_data")
    args.out = common.png_path_for(args.data, "fig2_score_comparison", args.out)
    d = np.load(args.data, allow_pickle=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- left: sorted score profiles ----
    for k in ["attention", "single_lookahead", "lorekv_multi"]:
        key = f"sorted_{k}"
        if key in d.files:
            y = _norm(d[key])
            ax1.plot(np.arange(len(y)), y, color=COLOR[k], lw=1.8, label=LABEL[k])
    ax1.axvline(int(d["budget"]), color="k", ls=":", lw=1.2, label=f"budget B={int(d['budget'])}")
    ax1.set_xlabel("Candidate rank (sorted by score)")
    ax1.set_ylabel("Normalized importance score")
    ax1.set_title("Importance-score profiles (representative layer)")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # ---- right: pairwise disagreement ----
    pairs = [p.replace("|", "\nvs\n") for p in d["pairs"]]
    x = np.arange(len(pairs))
    width = 0.38
    ax2.bar(x - width / 2, d["overlap_mean"], width, yerr=d["overlap_std"],
            capsize=3, color="#4c72b0", label=f"Top-B={int(d['budget'])} overlap")
    ax2.bar(x + width / 2, d["spearman_mean"], width, yerr=d["spearman_std"],
            capsize=3, color="#dd8452", label="Spearman rank corr.")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pairs, fontsize=7.5)
    ax2.set_ylabel("Agreement")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("How much the scoring rules disagree")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Raw score comparison  ({d['dataset']} LongBench)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
