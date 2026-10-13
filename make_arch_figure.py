"""LORE-KV architecture figure  ->  arch.png / arch.pdf

Draws the method pipeline of "Training-Free Test-Time Lookahead Causal KV Eviction"
(LORE-KV) exactly as it is implemented in this repository:

  1. Prefill              lorekv/monkeypatch/lookahead_runner.py, phase "collect"
                          -> full uncompressed K/V + prompt-window queries Q_W          (Eq. 1-2)
  2. Test-time lookahead  phase "lookahead": M continuations of length T, each on its own
                          cloned cache; post-RoPE query states kept as Q_look^(m)       (Eq. 1-2)
  3. Reliability weights  reliability_weights(): R_m = b*cbar_m - g*Hbar_m - l_D*Dhat_m,
                          omega = softmax(R)                                            (Eq. 3-4)
  4. Score + evict        kv_cluster.compress(): e = ||W_O do||^2, S_i = rho(E_W) + lam_L
                          sum_m omega_m rho(E_m), Alloc -> Select -> Gather, then decode
                          from (K*, V*) with the drafts discarded                       (Eq. 5-8)

Style: PALETTE / FigureStyle /
apply_publication_style / make_* helpers / finalize_figure, minimalist spines,
frameless legends, tight_layout finish, 300 dpi raster + vector PDF.

The only numbers in the figure are *illustrative* schematic values (particle weights,
candidate-score profile); they live in the ILLUSTRATIVE DATA block below and can be
replaced with measured arrays (e.g. `omega` from a capture .npz) without touching the
drawing code.

Usage
-----
    python make_arch_figure.py                  # writes ./arch.png and ./arch.pdf
    python make_arch_figure.py --out figures/arch --dpi 600
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / batch safe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Palette and style
# ---------------------------------------------------------------------------
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}
GREY_DARK = "#4D4D4D"
GREY_MID = "#767676"
INK = "#272727"
PANEL_FACE = "#F7F8FA"

# One colour per lookahead particle; reused for the strands (stage 2) and the
# weight bars (stage 3) so the reader can follow a particle across the figure.
PARTICLE_COLORS = [
    PALETTE["blue_secondary"],
    PALETTE["teal"],
    PALETTE["green_3"],
    PALETTE["violet"],
]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 15
    axes_linewidth: float = 2.0
    use_tex: bool = False
    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    """Configure rcParams once, before any figure is created."""
    style = style or FigureStyle()
    matplotlib.rcParams.update(
        {
            "font.family": list(style.font_family),
            "font.size": style.font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": INK,
            "text.color": INK,
            "legend.frameon": False,
            "mathtext.fontset": "dejavusans",
            "text.usetex": style.use_tex,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,  # editable (TrueType) text in the vector export
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return style


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    """Return (fig, axes) with axes always a flattened 1D array."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(np.asarray(axes)).ravel()


_VALID_FORMATS = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kwargs):
    """Save `fig` to every requested format next to `out_path` (parents created)."""
    out_path = Path(out_path)
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["pdf", "png"]
    bad = [f for f in formats if f.lower() not in _VALID_FORMATS]
    if bad:
        raise ValueError(f"unsupported export format(s): {bad}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    saved = []
    for fmt in formats:
        target = stem.with_suffix(f".{fmt.lower()}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kwargs)
        saved.append(target)
    if close:
        plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Diagram primitives (the make_* helpers this figure needs)
# ---------------------------------------------------------------------------
def make_panel(ax, x0, x1, y0, y1, index, title, face=PANEL_FACE, edge=PALETTE["neutral"]):
    """Rounded stage panel with a numbered badge and a bold title."""
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
            zorder=1,
        )
    )
    bx, by = x0 + 1.9, y1 - 2.6
    ax.add_patch(Circle((bx, by), 1.15, facecolor=PALETTE["blue_main"], edgecolor="none", zorder=3))
    ax.text(bx, by, str(index), ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", zorder=4)
    ax.text(bx + 2.0, by, title, ha="left", va="center", fontsize=13.5,
            fontweight="bold", color=INK, zorder=3)
    return (x0 + x1) / 2.0


def make_token_strip(ax, x0, y, n, cell=1.15, height=2.3, gap=0.20, colors=None,
                     edge=GREY_DARK, lw=0.9, alpha=1.0, hatch=None, zorder=3):
    """A row of KV/token cells. `colors` is a per-cell list (None -> neutral)."""
    colors = colors or [PALETTE["neutral"]] * n
    centers = []
    for i in range(n):
        x = x0 + i * (cell + gap)
        ax.add_patch(
            Rectangle((x, y), cell, height, facecolor=colors[i], edgecolor=edge,
                      linewidth=lw, alpha=alpha, hatch=hatch, zorder=zorder)
        )
        centers.append(x + cell / 2.0)
    return np.asarray(centers), x0 + n * (cell + gap) - gap


def make_flow_arrow(ax, x0, x1, y, label=None, color=GREY_DARK, lw=2.4, mutation=20,
                    label_dy=2.6, fontsize=10.5, style="-|>", label_color=None):
    ax.add_patch(
        FancyArrowPatch((x0, y), (x1, y), arrowstyle=style, mutation_scale=mutation,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=5)
    )
    if label:
        ax.text((x0 + x1) / 2.0, y + label_dy, label, ha="center", va="bottom",
                fontsize=fontsize, color=label_color or GREY_DARK, zorder=5)


def make_chip(ax, x, y, text, width=None, height=3.0, face="white", edge=PALETTE["neutral"],
              fontsize=11, color=INK, lw=1.3, ha="center"):
    """Small labelled rounded box used for tensors / equations."""
    if width is None:
        width = 0.62 * fontsize * 0.055 * len(text) + 2.0
    ax.add_patch(
        FancyBboxPatch((x - width / 2.0, y - height / 2.0), width, height,
                       boxstyle="round,pad=0,rounding_size=0.7", linewidth=lw,
                       edgecolor=edge, facecolor=face, zorder=3)
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=4)
    return width


def make_weight_bars(ax, x0, ys, values, height=2.3, max_len=11.0, colors=None,
                     fontsize=10, fmt="{:.2f}"):
    """Horizontal particle-weight bars, one per lookahead strand row."""
    colors = colors or PARTICLE_COLORS
    vmax = float(np.max(values))
    for y, v, c in zip(ys, values, colors):
        length = max_len * float(v) / vmax
        ax.add_patch(
            Rectangle((x0, y), length, height, facecolor=c, edgecolor=GREY_DARK,
                      linewidth=0.9, zorder=3)
        )
        ax.text(x0 + length + 0.6, y + height / 2.0, fmt.format(v), ha="left", va="center",
                fontsize=fontsize, color=GREY_DARK, zorder=4)


def make_score_profile(ax, x0, x1, y0, height, scores, budget, colors=("kept", "dropped"),
                       fontsize=10):
    """Candidate-score stems with a top-B cut, kept in blue / evicted in red."""
    n = len(scores)
    order = np.argsort(scores)[::-1]
    keep = np.zeros(n, dtype=bool)
    keep[order[:budget]] = True
    thresh = float(np.sort(scores)[::-1][budget - 1])

    span = (x1 - x0) / n
    bw = span * 0.62
    smax = float(scores.max()) * 1.12
    for i, s in enumerate(scores):
        h = height * s / smax
        ax.add_patch(
            Rectangle((x0 + i * span + (span - bw) / 2.0, y0), bw, h,
                      facecolor=PALETTE["blue_main"] if keep[i] else PALETTE["red_1"],
                      edgecolor=GREY_DARK if keep[i] else PALETTE["red_2"],
                      linewidth=0.6, zorder=3)
        )
    y_cut = y0 + height * thresh / smax
    ax.plot([x0 - 0.4, x1 + 0.4], [y_cut, y_cut], ls=(0, (4, 2.5)), lw=1.5,
            color=PALETTE["red_strong"], zorder=4)
    ax.text(x1 + 0.8, y_cut, "top-$B_{l,h}$", ha="left", va="center",
            fontsize=fontsize, color=PALETTE["red_strong"], zorder=4)
    return keep


def make_legend_swatches(ax, x, y, entries, dx=None, size=1.5, fontsize=10.5, gap=0.7):
    """Inline swatch legend: entries = [(color, label), ...]."""
    cursor = x
    for color, label in entries:
        ax.add_patch(
            Rectangle((cursor, y), size, size, facecolor=color, edgecolor=GREY_DARK,
                      linewidth=0.9, zorder=4)
        )
        ax.text(cursor + size + gap, y + size / 2.0, label, ha="left", va="center",
                fontsize=fontsize, color=GREY_DARK, zorder=4)
        cursor += size + gap + (dx if dx is not None else 0.60 * fontsize * 0.09 * len(label) + 2.4)
    return cursor


# ---------------------------------------------------------------------------
# ILLUSTRATIVE DATA  (schematic only -- swap in measured arrays if desired)
# ---------------------------------------------------------------------------
M = 4                  # cfg.num_particles
T = 6                  # cfg.lookahead_len (drawn length; the runs use T = 8-16)
W_SHOW = 3             # prompt-window cells highlighted as the observation window W
N_PROMPT_CELLS = 13    # prompt cells drawn
N_CANDIDATES = 34      # candidate cells in the score profile
KEEP = 13              # cells kept by top-B in the score profile

# omega_m = softmax(R_m), Eq. 3-4. Replace with real weights, e.g.
#   OMEGA = np.load("weighting_data.npz")["weights"]
OMEGA = np.array([0.38, 0.27, 0.21, 0.14])

_rng = np.random.default_rng(7)
SCORES = _rng.gamma(shape=1.7, scale=1.0, size=N_CANDIDATES) + 0.25
SCORES[[3, 11, 12, 21, 28]] += np.array([3.4, 4.2, 2.6, 3.9, 3.0])  # answer-bearing spikes


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def build_architecture_figure(figsize=(15.0, 6.4)):
    fig, axes = create_subplots(1, 1, figsize=figsize)
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 37.2)
    ax.set_axis_off()

    PY0, PY1 = 5.2, 35.4                  # panel top/bottom, shared by all stages
    SPANS = [(1.0, 21.5), (25.5, 47.0), (51.0, 69.0), (73.0, 99.0)]
    ROW_Y = [23.6, 20.2, 16.8, 13.4]      # the M particle rows, shared by stages 2 and 3

    # ---------------- Stage 1: prefill, no eviction ----------------
    x0, x1 = SPANS[0]
    make_panel(ax, x0, x1, PY0, PY1, 1, "Prefill, no eviction")

    ax.text((x0 + x1) / 2.0, 30.0, r"prompt  $X=(x_1,\dots,x_n)$", ha="center", va="center",
            fontsize=11.5, color=GREY_DARK)
    prompt_colors = (
        [PALETTE["highlight"]]
        + [PALETTE["neutral"]] * (N_PROMPT_CELLS - 1 - W_SHOW)
        + [PALETTE["highlight"]] * W_SHOW
    )
    strip_w = N_PROMPT_CELLS * (1.15 + 0.20) - 0.20
    sx0 = (x0 + x1) / 2.0 - strip_w / 2.0
    centers, _ = make_token_strip(ax, sx0, 26.2, N_PROMPT_CELLS, colors=prompt_colors)
    ax.text(centers[0], 25.4, "sink", ha="center", va="top", fontsize=9, color=GREY_MID)
    ax.text(centers[-2], 25.4, r"window $\mathcal{W}$", ha="center", va="top",
            fontsize=9, color=GREY_MID)

    ax.add_patch(FancyArrowPatch(((x0 + x1) / 2.0, 23.4), ((x0 + x1) / 2.0, 21.4),
                                 arrowstyle="-|>", mutation_scale=16, linewidth=2.0,
                                 color=GREY_DARK, zorder=5))
    make_chip(ax, (x0 + x1) / 2.0, 19.4, r"frozen LLM  $f_\theta$", width=16.0, height=3.4,
              face="white", edge=PALETTE["blue_main"], fontsize=12.5, lw=1.8)
    ax.add_patch(FancyArrowPatch(((x0 + x1) / 2.0, 17.6), ((x0 + x1) / 2.0, 15.6),
                                 arrowstyle="-|>", mutation_scale=16, linewidth=2.0,
                                 color=GREY_DARK, zorder=5))
    make_chip(ax, (x0 + x1) / 2.0, 13.6, r"full cache  $\mathbf{K},\mathbf{V}$",
              width=16.0, height=3.0, face=PALETTE["green_1"], edge=PALETTE["green_3"],
              fontsize=11.5)
    make_chip(ax, (x0 + x1) / 2.0, 9.9, r"window queries  $\mathcal{Q}_{\mathcal{W}}$",
              width=16.0, height=3.0, face=PALETTE["green_1"], edge=PALETTE["green_3"],
              fontsize=11.5)
    ax.text((x0 + x1) / 2.0, 6.8,
            r"protected $\mathcal{P}=\{1\}\cup\mathcal{W}\cup\mathcal{B}$,  scored $\mathcal{C}$",
            ha="center", va="center", fontsize=9.5, color=GREY_MID)

    # ---------------- Stage 2: test-time lookahead ----------------
    x0, x1 = SPANS[1]
    make_panel(ax, x0, x1, PY0, PY1, 2, "Test-time lookahead")
    ax.text((x0 + x1) / 2.0, 30.0,
            r"$Y^{(m)}\sim f_\theta(\cdot\,|\,X)$,   $m=1,\dots,M$,   $|Y^{(m)}|=T$",
            ha="center", va="center", fontsize=10.5, color=GREY_DARK)

    strand_x0 = x0 + 5.6
    for m, (y, c) in enumerate(zip(ROW_Y, PARTICLE_COLORS)):
        ax.text(strand_x0 - 1.0, y + 1.15, rf"$Y^{{({m + 1})}}$", ha="right", va="center",
                fontsize=10.5, color=c)
        make_token_strip(ax, strand_x0, y, T, cell=1.5, height=2.3, gap=0.28,
                         colors=[c] * T, alpha=0.92)
    ax.plot([x1 - 3.2, x1 - 3.2], [ROW_Y[-1], ROW_Y[0] + 2.3], lw=1.6, color=GREY_MID, zorder=3)
    ax.text(x1 - 2.6, (ROW_Y[-1] + ROW_Y[0] + 2.3) / 2.0,
            r"post-RoPE  $\mathcal{Q}^{(m)}_{\mathrm{look}}$", rotation=90,
            ha="left", va="center", fontsize=10.5, color=GREY_DARK)
    make_chip(ax, (x0 + x1) / 2.0, 9.6,
              r"$\mathcal{Q}_{\mathrm{score}}=\mathcal{Q}_{\mathcal{W}}\cup\bigcup_m\mathcal{Q}^{(m)}_{\mathrm{look}}$",
              width=19.0, height=3.6, face="white", edge=PALETTE["blue_secondary"], fontsize=12)
    ax.text((x0 + x1) / 2.0, 6.6, "each particle on its own cloned cache",
            ha="center", va="center", fontsize=9.5, color=GREY_MID)

    # ---------------- Stage 3: reliability weighting ----------------
    x0, x1 = SPANS[2]
    make_panel(ax, x0, x1, PY0, PY1, 3, "Reliability weights")
    ax.text((x0 + x1) / 2.0, 30.2,
            r"$R_m=\beta\,\bar{c}_m-\gamma\,\bar{H}_m-\lambda_D\hat{D}_m$",
            ha="center", va="center", fontsize=12, color=INK)
    ax.text((x0 + x1) / 2.0, 27.0, r"$\omega=\mathrm{softmax}(\mathbf{R})$",
            ha="center", va="center", fontsize=12, color=INK)
    make_weight_bars(ax, x0 + 2.4, ROW_Y, OMEGA, max_len=10.0)
    ax.text((x0 + x1) / 2.0, 9.6,
            "atypicality  ·  entropy  ·  disagreement", ha="center", va="center",
            fontsize=9.5, color=GREY_DARK)
    ax.text((x0 + x1) / 2.0, 6.8,
            r"/E: $(1,1.3,0)$    /D: $(1,1,0.3)$", ha="center", va="center",
            fontsize=9.5, color=GREY_MID)

    # ---------------- Stage 4: projected deletion score + eviction ----------------
    x0, x1 = SPANS[3]
    make_panel(ax, x0, x1, PY0, PY1, 4, "Score, then evict")
    ax.text((x0 + x1) / 2.0, 30.8,
            r"$e_{q,i}=\|\,\mathbf{W}^{O}\Delta\mathbf{o}_{q,i}\|_2^2$,   "
            r"$\Delta\mathbf{o}_{q,i}\!\approx\!\frac{a_{q,i}}{1-a_{q,i}}"
            r"(\mathbf{v}_i-\mathbf{o}_q)$",
            ha="center", va="center", fontsize=10.5, color=INK)
    ax.text((x0 + x1) / 2.0, 26.8,
            r"$S_i=\rho(\mathcal{E}_{i,\mathcal{W}})+\lambda_L\sum_m\omega_m\,\rho(\mathcal{E}_{i,m})$",
            ha="center", va="center", fontsize=12, color=INK)

    sx0, sx1 = x0 + 3.2, x1 - 6.0
    ax.plot([sx0 - 0.9, sx0 - 0.9], [15.4, 23.6], lw=1.6, color=GREY_DARK, zorder=3)
    ax.text(sx0 - 1.8, 19.5, r"$S_i$", rotation=90, ha="center", va="center",
            fontsize=11, color=GREY_DARK)
    keep = make_score_profile(ax, sx0, sx1, 15.4, 8.2, SCORES, KEEP)
    ax.text((sx0 + sx1) / 2.0, 14.6, r"candidates $i\in\mathcal{C}$", ha="center", va="top",
            fontsize=9.5, color=GREY_MID)

    ax.add_patch(FancyArrowPatch(((sx0 + sx1) / 2.0, 13.6), ((sx0 + sx1) / 2.0, 11.9),
                                 arrowstyle="-|>", mutation_scale=16, linewidth=2.0,
                                 color=GREY_DARK, zorder=5))
    ax.text((sx0 + sx1) / 2.0 + 0.9, 12.7,
            r"$\mathrm{Alloc}\rightarrow\mathrm{Select}\rightarrow\mathrm{Gather}$",
            ha="left", va="center", fontsize=10, color=GREY_DARK)

    cache_cells = [PALETTE["highlight"]] * 2 + [
        PALETTE["blue_main"] if k else "white" for k in keep[:14]
    ]
    n_cells = len(cache_cells)
    cw, cgap = 1.15, 0.20
    cx0 = (x0 + x1) / 2.0 - (n_cells * (cw + cgap) - cgap) / 2.0
    for i, c in enumerate(cache_cells):
        cx = cx0 + i * (cw + cgap)
        dropped = c == "white"
        ax.add_patch(
            Rectangle((cx, 8.4), cw, 2.3, facecolor="white" if dropped else c,
                      edgecolor=PALETTE["red_2"] if dropped else GREY_DARK,
                      linewidth=0.9, linestyle=(0, (2, 1.6)) if dropped else "solid",
                      zorder=3)
        )
    ax.text((x0 + x1) / 2.0, 6.7,
            r"compressed cache $\mathbf{K}^{*},\mathbf{V}^{*}$  —  drafts discarded",
            ha="center", va="center", fontsize=10.5, color=GREY_DARK)

    # ---------------- Inter-stage flow ----------------
    arrow_y = 19.0
    labels = [
        r"$\mathbf{K},\mathbf{V}$,  $\mathcal{Q}_{\mathcal{W}}$",
        r"$\bar{c}_m,\bar{H}_m,\hat{D}_m$",
        r"$\omega_m$",
    ]
    for (a, b), lab in zip(zip([s[1] for s in SPANS[:-1]], [s[0] for s in SPANS[1:]]), labels):
        make_flow_arrow(ax, a + 0.5, b - 0.5, arrow_y, label=lab, label_dy=1.6,
                        color=PALETTE["blue_main"], lw=2.6, mutation=22, fontsize=10.5,
                        label_color=PALETTE["blue_main"])

    # ---------------- Footer banner + swatch legend ----------------
    ax.add_patch(
        FancyBboxPatch((1.0, 0.6), 98.0, 3.4, boxstyle="round,pad=0,rounding_size=1.0",
                       linewidth=1.4, edgecolor=PALETTE["blue_secondary"],
                       facecolor="#EEF3F9", zorder=1)
    )
    ax.text(3.0, 2.3,
            r"training-free  ·  frozen $\theta$  ·  one-time $M\times T$ lookahead decode before "
            r"generation  ·  steady-state memory and per-token decode unchanged",
            ha="left", va="center", fontsize=11, color=PALETTE["blue_main"], zorder=3)
    make_legend_swatches(
        ax, 68.0, 35.9,
        [
            (PALETTE["highlight"], "protected"),
            (PALETTE["blue_main"], "retained"),
            (PALETTE["red_1"], "evicted"),
        ],
        fontsize=10.5,
    )

    fig.tight_layout(pad=1.0)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="arch", help="output basename (default: ./arch)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()

    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))
    fig = build_architecture_figure()
    for path in finalize_figure(fig, args.out, formats=args.formats, dpi=args.dpi, pad=0.06):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
