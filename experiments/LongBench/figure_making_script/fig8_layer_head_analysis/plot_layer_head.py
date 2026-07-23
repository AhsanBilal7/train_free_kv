"""Figure 8 (plot stage) - reads layer_head_data.npz.

Left  : layer perturbation energy E_l across depth.
Right : heatmap of per-(layer, head) future-query recall gain G_lh -> shows the gain
        is concentrated in specific heads/layers, motivating adaptive allocation.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--heatmap", default="gain", choices=["gain", "mismatch"])
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "layer_head_data")
    args.out = common.png_path_for(args.data, "fig8_layer_head_analysis", args.out)
    d = np.load(args.data, allow_pickle=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    E = d["E_layer"]
    ax1.plot(np.arange(len(E)), E, marker="o", ms=3, lw=1.6, color="#d62728")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Perturbation energy $E_l=\\sum_h\\sum_i S_i^{l,h}$")
    ax1.set_title("Layer-wise perturbation energy")
    ax1.grid(True, alpha=0.3)

    mat = d["G_head"] if args.heatmap == "gain" else d["MM_head"]
    title = ("Head future-query recall gain $G_{l,h}$" if args.heatmap == "gain"
             else "Head query mismatch $MM_{l,h}$")
    cmap = "RdBu_r" if args.heatmap == "gain" else "viridis"
    vmax = np.nanmax(np.abs(mat)) if args.heatmap == "gain" else np.nanmax(mat)
    vmin = -vmax if args.heatmap == "gain" else 0.0
    im = ax2.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax2.set_xlabel("KV head index")
    ax2.set_ylabel("Layer index")
    ax2.set_title(title)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f"Layer/head structure of LORE-KV scores (B={int(d['budget'])}, {d['dataset']})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
