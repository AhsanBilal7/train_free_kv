"""Figure 5 (plot stage) - reads calibration_data.npz.

A  calibration curve  : bin particles by predicted confidence, plot mean predicted
                        confidence vs mean observed utility.
B  risk-coverage       : keep the most-confident fraction of particles, plot that
                        coverage vs mean alignment error of the kept particles.
C  correlation bars    : Spearman(signal, oracle utility) for log-prob, -entropy,
                        combined R_m, the normalized weight, and the trajectory
                        disagreement -D_m -- the critical ablation that isolates
                        what each second term of R_m actually adds.
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


def _minmax(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--utility", default="recall", choices=["recall", "align"])
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "calibration_data")
    args.out = common.png_path_for(args.data, "fig5_confidence_calibration", args.out)
    d = np.load(args.data, allow_pickle=True)

    util = d["util_recall"] if args.utility == "recall" else d["util_align"]
    conf = _minmax(d["combined"])          # predicted confidence (normalized R_m)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))

    # A: calibration
    order = np.argsort(conf)
    bins = np.array_split(order, args.n_bins)
    xp = [conf[b].mean() for b in bins]
    yp = [util[b].mean() for b in bins]
    ye = [util[b].std() for b in bins]
    ax1.errorbar(xp, yp, yerr=ye, marker="o", lw=1.8, color="#d62728", capsize=3)
    ax1.set_xlabel("Predicted confidence (normalized $R_m$)")
    ax1.set_ylabel(f"Observed utility ({args.utility})")
    ax1.set_title("Calibration")
    ax1.grid(True, alpha=0.3)

    # B: risk-coverage (keep most-confident fraction)
    desc = np.argsort(-d["combined"])
    fracs = np.linspace(0.1, 1.0, 10)
    err = []
    for fr in fracs:
        k = max(1, int(fr * len(desc)))
        kept = desc[:k]
        err.append(1.0 - d["util_align"][kept].mean())
    ax2.plot(fracs, err, marker="s", lw=1.8, color="#1f77b4")
    ax2.set_xlabel("Fraction of particles retained (most confident first)")
    ax2.set_ylabel("Mean alignment error (1 - align)")
    ax2.set_title("Risk-coverage")
    ax2.grid(True, alpha=0.3)

    # C: correlation of each signal with utility
    signals = {
        "log-prob": d["logprob"],
        "-entropy": -d["entropy"],
        "combined $R_m$": d["combined"],
        "weight $\\omega_m$": d["weight"],
    }
    # the term that replaces entropy in R_m (older .npz files predate it)
    if "divergence_z" in d:
        signals["$-\\hat{D}_m$ (disagr.)"] = -d["divergence_z"]
    names = list(signals)
    corr = [_spearman(signals[k], util) for k in names]
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(names)]
    ax3.bar(np.arange(len(names)), corr, color=colors)
    ax3.set_xticks(np.arange(len(names)))
    ax3.set_xticklabels(names, fontsize=8)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_ylabel(f"Spearman(signal, {args.utility})")
    ax3.set_title("Which signal predicts utility")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Confidence-weight calibration (B={int(d['budget'])}, {d['dataset']})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
