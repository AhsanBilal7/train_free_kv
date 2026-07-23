"""Figure 3 (plot stage) - reads distribution_data.npz.

Left  : distribution of decode-time attention-output cosine similarity to the FULL
        cache, for LORE-KV eviction vs attention-only eviction (mass near 1 = the
        evicted cache reproduces the full-cache output distribution).
Right : that fidelity across model depth (per-layer mean cosine).
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
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "distribution_data")
    args.out = common.png_path_for(args.data, "fig3_cache_distribution", args.out)
    d = np.load(args.data, allow_pickle=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    bins = np.linspace(-0.2, 1.0, 61)
    ax1.hist(d["cos_lorekv"], bins=bins, density=True, alpha=0.55, color="#d62728",
             label=f"LORE-KV  (mean {d['cos_lorekv'].mean():.3f})")
    ax1.hist(d["cos_attn"], bins=bins, density=True, alpha=0.55, color="#7f7f7f",
             label=f"Attention-only  (mean {d['cos_attn'].mean():.3f})")
    ax1.set_xlabel("cos( evicted-cache output , full-cache output )")
    ax1.set_ylabel("Density")
    ax1.set_title("Decode-time output fidelity distribution")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    plc = d["per_layer_cos_lorekv"]; pca = d["per_layer_cos_attn"]
    x = np.arange(len(plc))
    ax2.plot(x, plc, color="#d62728", marker="o", ms=3, lw=1.6, label="LORE-KV")
    ax2.plot(x, pca, color="#7f7f7f", marker="s", ms=3, lw=1.6, label="Attention-only")
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Mean output cosine to full cache")
    ax2.set_title("Output fidelity across depth")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Full-cache vs evicted-cache output distribution "
                 f"(B={int(d['budget'])}, {d['dataset']})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
