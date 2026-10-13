"""Gain against query mismatch, plot stage (paper Figure 6).

Scatter of the mismatch of each sample (x) against the gain of LORE-KV (y), with a
regression line and the Spearman correlation.
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


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else 0.0


def _panel(ax, x, y, ylabel, title):
    ax.scatter(x, y, s=26, alpha=0.6, color="#1f77b4", edgecolor="none")
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, color="#d62728", lw=2,
                label=f"slope={m:.2f}, $\\rho$={_spearman(x, y):.2f}")
        ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Prompt-vs-future query mismatch  $M(X)=1-$align$(Q_W, Q_{true})$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "mismatch_data")
    args.out = common.png_path_for(args.data, "fig6_mismatch_vs_gain", args.out)
    d = np.load(args.data, allow_pickle=True)
    x = d["mismatch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    _panel(ax1, x, d["gain_recall"], "Oracle recall@B gain\n(LORE-KV − window-only)",
           "Retention-recall gain vs mismatch")
    _panel(ax2, x, d["gain_fid_vs_attn"], "Output-fidelity gain\n(LORE-KV − attention)",
           "Output-fidelity gain vs mismatch")
    fig.suptitle(f"Gain grows with query mismatch (B={int(d['budget'])}, {d['dataset']})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
