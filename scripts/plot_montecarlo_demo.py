"""Draw the M lookahead trajectories as Monte Carlo samples of answer-time queries (paper Figure 1).

The script reads the output of ``scripts/demo_montecarlo_queries.py`` for one prompt
and draws:

(a) the query states in 2-D (PCA of their union): the prompt window (grey), the M
    trajectories (colors) and the true answer-time queries (black stars);
(b) the coverage of the true queries as a function of M, over nested prefixes of
    the same sample, with the prompt-window value as reference;
(c) the trajectories as text, with their weight and coverage, and the real answer.

Usage::

    python scripts/plot_montecarlo_demo.py                       # newest capture
    python scripts/plot_montecarlo_demo.py --run demos/montecarlo/<tag> --sample 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
GREY_DARK, GREY_MID, INK = "#4D4D4D", "#767676", "#272727"
PARTICLE_COLORS = [PALETTE["blue_secondary"], PALETTE["teal"], PALETTE["green_3"],
                   PALETTE["violet"], "#E8820C", "#7B5FB0", "#17A2B8", "#C7302B"]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 15
    axes_linewidth: float = 2.0
    font_family: tuple = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None):
    style = style or FigureStyle()
    matplotlib.rcParams.update({
        "font.family": list(style.font_family), "font.size": style.font_size,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
        "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return style


_VALID = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kw):
    out_path = Path(out_path)
    formats = formats or ["pdf", "png"]
    bad = [f for f in formats if f.lower() not in _VALID]
    if bad:
        raise ValueError(f"unsupported export format(s): {bad}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem, saved = out_path.with_suffix(""), []
    for fmt in formats:
        t = stem.with_suffix(f".{fmt.lower()}")
        fig.savefig(t, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kw)
        saved.append(t)
    if close:
        plt.close(fig)
    return saved


def newest_run(root):
    runs = [d for d in glob.glob(os.path.join(root, "*")) if glob.glob(os.path.join(d, "*.npz"))]
    if not runs:
        raise SystemExit(f"no captures under {root}; run scripts/demo_montecarlo_queries.py first")
    return max(runs, key=os.path.getmtime)


def make_query_scatter(ax, d, colors):
    """Draw the PCA of the query states: window, trajectories and true decoding steps."""
    coords = np.asarray(d["coords"])
    nW, M, T = int(d["n_window"]), int(d["M"]), int(d["T"])
    W, P, F = coords[:nW], coords[nW:nW + M * T], coords[nW + M * T:]

    ax.scatter(W[:, 0], W[:, 1], s=52, marker="x", linewidths=1.8,
               color=GREY_MID, label=r"prompt window $\mathcal{Q}_\mathcal{W}$", zorder=2)
    for m in range(M):
        pm = P[m * T:(m + 1) * T]
        ax.plot(pm[:, 0], pm[:, 1], "-", lw=1.0, color=colors[m % len(colors)],
                alpha=0.55, zorder=3)
        ax.scatter(pm[:, 0], pm[:, 1], s=46, color=colors[m % len(colors)],
                   edgecolor="white", linewidth=0.6, zorder=4,
                   label=r"sampled $\mathcal{Q}^{(m)}_{\mathrm{look}}$" if m == 0 else None)
    ax.scatter(F[:, 0], F[:, 1], s=150, marker="*", color="black", zorder=5,
               label=r"true $\mathcal{Q}_{\mathrm{future}}$")
    ax.set_xlabel(f"PC 1     (PC1+PC2 = {100 * float(d['evr']):.0f}\\% of variance)"
                  .replace("\\%", "%"))
    ax.set_ylabel("PC 2")
    ax.set_title(f"(a)  Query states, layer {int(d['layer'])}, head {int(d['head'])}\n"
                 f"coverage {float(d['cov_window']):.3f} $\\to$ "
                 f"{float(d['cov_lookahead']):.3f}", fontsize=15, loc="left")
    ax.legend(fontsize=11, loc="best")


def make_coverage_curve(ax, d):
    cov = np.asarray(d["cov_vs_m"], dtype=float)
    m = np.arange(1, len(cov) + 1)
    ax.plot(m, cov, "-o", color=PALETTE["blue_main"], lw=2.2, ms=8)
    base = float(d["cov_window"])
    ax.axhline(base, ls=(0, (4, 3)), lw=1.8, color=GREY_MID)
    ax.text(len(cov), base, r" prompt window $\mathcal{Q}_\mathcal{W}$", color=GREY_MID,
            fontsize=11, va="bottom", ha="right")
    ax.set_xlabel(r"trajectories $M$")
    ax.set_ylabel(r"coverage of $\mathcal{Q}_{\mathrm{future}}$")
    ax.set_xticks(m)
    lo = min(cov.min(), float(d["cov_window"]))
    hi = cov.max()
    ax.set_ylim(lo - 0.16 * (hi - lo + 1e-6), hi + 0.10 * (hi - lo + 1e-6))
    ax.set_title("(b)  The estimator converging", fontsize=15, loc="left")


def make_text_panel(ax, texts, d, colors, width=54, max_rows=5):
    """Show the trajectories as text with their weight and coverage, and the real answer.

    Equal continuations are shown one time with a count. The trajectory with the
    lowest weight is always shown.
    """
    ax.set_axis_off()
    parts = texts["particles"]
    y, dy = 1.0, 0.062
    ax.text(0, y, r"$\bf{prompt\ ends:}$  " + "…" + textwrap.shorten(
        texts["prompt_tail"].replace("\n", " "), width=width + 8, placeholder="…"),
        fontsize=11, color=GREY_DARK, va="top", transform=ax.transAxes)
    y -= dy * 1.5
    ax.text(0, y, r"$\bf{M\ sampled\ continuations}$  "
                  r"($\omega_m$ = reliability weight, cov = coverage of $\mathcal{Q}_{\mathrm{future}}$)",
            fontsize=11.5, color=INK, va="top", transform=ax.transAxes)
    y -= dy

    groups = {}
    for p in parts:
        groups.setdefault(p["text"].strip(), []).append(p)
    groups = sorted(groups.values(), key=lambda g: -max(x["omega"] for x in g))
    lowest = min(parts, key=lambda p: p["omega"])
    shown, rows = set(), []
    for g in groups:
        if len(rows) >= max_rows - 1 and any(lowest in gg for gg in [g]) is False:
            continue
        rows.append(g); shown.update(id(x) for x in g)
        if len(rows) >= max_rows:
            break
    if id(lowest) not in shown:                      # force the outlier in
        rows = rows[:max_rows - 1] + [[lowest]]

    n_listed = sum(len(g) for g in rows)
    for g in rows:
        p = max(g, key=lambda x: x["omega"])
        col = colors[p["m"] % len(colors)]
        line = textwrap.shorten(p["text"].replace("\n", " ") or "(empty)",
                                width=width, placeholder="…")
        mult = f"  ($\\times${len(g)})" if len(g) > 1 else ""
        flag = p is lowest and len(parts) > 1
        ax.text(0.005, y, "▬", color=col, fontsize=13, va="top", transform=ax.transAxes)
        ax.text(0.045, y, f"$Y^{{({p['m']+1})}}$  “{line}”{mult}", fontsize=11, va="top",
                color=PALETTE["red_strong"] if flag else INK, transform=ax.transAxes)
        ax.text(1.0, y, f"$\\omega$={p['omega']:.2f}  cov={p['coverage']:.3f}",
                fontsize=10.5, va="top", ha="right",
                color=PALETTE["red_strong"] if flag else GREY_DARK, transform=ax.transAxes)
        y -= dy
        if flag:
            ax.text(0.045, y, "degenerate branch, smallest weight $\\omega$",
                    fontsize=10, va="top", color=PALETTE["red_strong"],
                    transform=ax.transAxes)
            y -= dy * 0.85
    if n_listed < len(parts):
        ax.text(0.045, y, f"… {len(parts) - n_listed} further trajectories",
                fontsize=10.5, va="top", color=GREY_MID, transform=ax.transAxes)
        y -= dy
    y -= dy * 0.4
    ax.text(0, y, r"$\bf{what\ the\ model\ actually\ answers}$ (full cache, greedy):",
            fontsize=11.5, color=INK, va="top", transform=ax.transAxes)
    y -= dy
    ax.text(0.045, y, "“" + textwrap.shorten(texts["true_continuation"].replace("\n", " "),
                                             width=width + 6, placeholder="…") + "”",
            fontsize=11, va="top", color=PALETTE["blue_main"], transform=ax.transAxes)
    if texts.get("answers"):
        y -= dy
        ax.text(0.045, y, "gold: " + textwrap.shorten(
            "; ".join(texts["answers"]).replace("\n", " "), width=width, placeholder="…"),
            fontsize=10.5, va="top", color=GREY_DARK, transform=ax.transAxes)
    ax.set_title("(c)  What one Monte-Carlo draw looks like", fontsize=15, loc="left")


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, help="capture folder (default: newest)")
    ap.add_argument("--root", default=os.path.join(repo, "demos", "montecarlo"))
    ap.add_argument("--sample", default=None, help="substring picking the sample file")
    ap.add_argument("--with-text", action="store_true",
                    help="add the third panel listing the sampled continuations")
    ap.add_argument("--out", default=None, help="output basename (default: <run>/montecarlo_queries)")
    args = ap.parse_args()

    run = args.run or newest_run(args.root)
    npzs = sorted(glob.glob(os.path.join(run, "*.npz")))
    if args.sample:
        npzs = [f for f in npzs if args.sample in os.path.basename(f)] or npzs
    npz = npzs[0]
    d = np.load(npz, allow_pickle=True)
    with open(npz.replace(".npz", ".json")) as fh:
        texts = json.load(fh)

    apply_publication_style()
    if args.with_text:
        fig = plt.figure(figsize=(19, 6.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.78, 1.25], wspace=0.28)
        make_query_scatter(fig.add_subplot(gs[0, 0]), d, PARTICLE_COLORS)
        make_coverage_curve(fig.add_subplot(gs[0, 1]), d)
        make_text_panel(fig.add_subplot(gs[0, 2]), texts, d, PARTICLE_COLORS)
    else:
        # Plots only, as in the MATLAB figure: the continuations are quoted in the
        # caption instead, where they are typeset by LaTeX proper.
        fig = plt.figure(figsize=(13, 5.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.9], wspace=0.24)
        make_query_scatter(fig.add_subplot(gs[0, 0]), d, PARTICLE_COLORS)
        make_coverage_curve(fig.add_subplot(gs[0, 1]), d)

    out = args.out or os.path.join(run, "montecarlo_queries")
    for p in finalize_figure(fig, out, formats=["png", "pdf"], dpi=300, pad=0.06):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
