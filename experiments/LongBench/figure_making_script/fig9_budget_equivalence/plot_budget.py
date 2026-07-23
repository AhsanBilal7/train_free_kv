"""Figure 9 (plot stage) - reads budget_data.npz.

Left  : fidelity F(B) vs budget for each method.
Right : memory saving of LORE-KV vs each baseline: for each LORE-KV budget B, the
        smallest baseline budget B_eq matching F_LORE(B), plotted as
        saving = 1 - B_eq/B (%). Baseline curves are interpolated for the inversion.
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


def _equiv_budget(target_F, base_budgets, base_F):
    """Smallest baseline budget whose fidelity >= target_F (linear interpolation,
    monotone-increasing assumption). Returns np.nan if even the largest can't match."""
    order = np.argsort(base_budgets)
    bb = np.array(base_budgets)[order]; bf = np.array(base_F)[order]
    if target_F <= bf[0]:
        return float(bb[0])
    for i in range(1, len(bb)):
        if bf[i] >= target_F:
            f0, f1 = bf[i - 1], bf[i]
            b0, b1 = bb[i - 1], bb[i]
            if f1 == f0:
                return float(b1)
            return float(b0 + (b1 - b0) * (target_F - f0) / (f1 - f0))
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "budget_data")
    args.out = common.png_path_for(args.data, "fig9_budget_equivalence", args.out)
    d = np.load(args.data, allow_pickle=True)
    budgets = d["budgets"]; methods = [str(m) for m in d["methods"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for m in methods:
        ax1.plot(budgets, d[f"fidelity_{m}"], marker="o", ms=4, lw=1.8,
                 color=COLOR.get(m), label=LABEL.get(m, m))
    ax1.set_xscale("log", base=2); ax1.set_xticks(budgets)
    ax1.set_xticklabels([str(int(b)) for b in budgets])
    ax1.set_xlabel("KV budget B"); ax1.set_ylabel("Full-cache output fidelity F(B)")
    ax1.set_title("Behavioral fidelity vs budget")
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)

    lore_F = d["fidelity_lorekv_multi"]
    for m in methods:
        if m == "lorekv_multi":
            continue
        savings = []
        for B, tF in zip(budgets, lore_F):
            beq = _equiv_budget(tF, budgets, d[f"fidelity_{m}"])
            savings.append(np.nan if np.isnan(beq) else 100.0 * (1.0 - B / beq))
        ax2.plot(budgets, savings, marker="s", ms=5, lw=1.8, color=COLOR.get(m),
                 label=f"vs {LABEL.get(m, m)}")
    ax2.axhline(0, color="k", lw=0.8, ls=":")
    ax2.set_xscale("log", base=2); ax2.set_xticks(budgets)
    ax2.set_xticklabels([str(int(b)) for b in budgets])
    ax2.set_xlabel("LORE-KV budget B")
    ax2.set_ylabel("Memory saving at equal fidelity (%)")
    ax2.set_title("Budget-equivalence saving")
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    fig.suptitle(f"Budget-equivalence / memory saving ({d['dataset']})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
