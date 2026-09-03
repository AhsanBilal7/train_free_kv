"""Figure 1 (plot stage) - reads alignment_data.npz and draws the alignment curves.

  x-axis : number of lookahead particles M
  y-axis : cosine coverage of the true full-cache decode queries
  curves : prompt-window / single trajectory, then one per (beta, gamma, lambda_D)
           cell of the Eq. 3 weighting grid (uniform / log-prob / typicality /
           entropy / disagreement / disagreement-instead-of-entropy)
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
STYLE = {
    "prompt_window": ("Prompt-window queries", "#7f7f7f", "o", "--"),
    "single_traj":   ("Single trajectory (M=1)", "#1f77b4", "s", "--"),
    # Labels are the (beta, gamma, lambda_D) coordinates of Eq. 3. Note that the
    # draft's "log-probability only" is (1, 1, 0) -- a FULL unit of entropy penalty --
    # while (1, 0, 0) is the actual no-entropy control.
    "uniform":       (r"Uniform  $(0,0,0)$", "#2ca02c", "^", "-"),
    "logprob":       (r"Log-prob  $(1,1,0)$", "#ff7f0e", "D", "-"),
    "typicality":    (r"Typicality  $(1,0,0)$", "#9467bd", "v", "-"),
    "entropy":       (r"Entropy (/E)  $(1,\gamma,0)$", "#1f77b4", "*", "-"),
    "divergence":    (r"Disagreement (/D)  $(1,1,\lambda_D)$", "#8c564b", "P", "-"),
    "typ_div":       (r"Disagreement, no entropy  $(1,0,\lambda_D)$", "#d62728", "X", "-"),
    # legacy keys from pre-reparametrization .npz files
    "lorekv":        (r"Log-prob $-$ entropy (legacy)", "#1f77b4", "*", ":"),
    "lorekv_div":    (r"Log-prob $-$ disagreement (legacy)", "#d62728", "P", ":"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "alignment_data")
    args.out = common.png_path_for(args.data, "fig1_future_query_alignment", args.out)

    d = np.load(args.data, allow_pickle=True)
    m_list = d["m_list"]
    n_samples = int(d["n_samples"])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for key, (label, color, marker, ls) in STYLE.items():
        if f"mean_{key}" not in d:
            continue          # .npz predates this strategy (or is a legacy capture)
        mean = d[f"mean_{key}"]
        std = d[f"std_{key}"]
        # prompt-window & single-trajectory are M-independent -> draw flat reference lines
        if key in ("prompt_window", "single_traj"):
            y = float(np.nanmean(mean))
            ax.axhline(y, color=color, ls=ls, lw=1.4, alpha=0.9, label=label)
        else:
            ax.plot(m_list, mean, color=color, marker=marker, ls=ls, lw=1.8, ms=7, label=label)
            ax.fill_between(m_list, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xscale("log", base=2)
    ax.set_xticks(m_list)
    ax.set_xticklabels([str(int(m)) for m in m_list])
    ax.set_xlabel("Number of lookahead particles $M$")
    ax.set_ylabel("Future-query cosine coverage")
    ax.set_title(f"Approximating the true decoding queries\n"
                 f"(budget B={int(d['budget'])}, {d['dataset']}, {n_samples} samples)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
