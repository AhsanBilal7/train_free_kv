"""Retention by token class, plot stage (paper Figure 11).

Left: retention rate of each class for the three scores. Right: enrichment
P(class | kept) / P(class | prompt); a value above 1 means the class is kept more
often than its share of the prompt.
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
LABEL = {"attention": "Attention (SnapKV)", "andpro_window": "AnDPro-window", "lorekv_multi": "LORE-KV"}
COLOR = {"attention": "#7f7f7f", "andpro_window": "#1f77b4", "lorekv_multi": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "composition_data")
    args.out = common.png_path_for(args.data, "fig7_retained_token_composition", args.out)
    d = np.load(args.data, allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    methods = [str(m) for m in d["methods"]]
    x = np.arange(len(classes))
    w = 0.8 / len(methods)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, m in enumerate(methods):
        ax1.bar(x + (j - (len(methods) - 1) / 2) * w, d[f"retention_{m}"], w,
                color=COLOR.get(m, None), label=LABEL.get(m, m))
    ax1.set_xticks(x); ax1.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Retention rate (mean fraction of heads)")
    ax1.set_title("What each method keeps, by token class")
    ax1.grid(True, axis="y", alpha=0.3); ax1.legend(fontsize=8)

    for j, m in enumerate(methods):
        ax2.bar(x + (j - (len(methods) - 1) / 2) * w, d[f"enrichment_{m}"], w,
                color=COLOR.get(m, None), label=LABEL.get(m, m))
    ax2.axhline(1.0, color="k", lw=0.8, ls=":")
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Enrichment  P(class|retained)/P(class|prompt)")
    ax2.set_title("Over- / under-retention (>1 = enriched)")
    ax2.grid(True, axis="y", alpha=0.3); ax2.legend(fontsize=8)

    fig.suptitle(f"Retained-token semantic composition (B={int(d['budget'])}, {d['dataset']})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
