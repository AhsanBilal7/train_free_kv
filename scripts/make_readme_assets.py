#!/usr/bin/env python3
"""Make the README and project-page figures in ``assets/``.

Usage::

    python scripts/make_readme_assets.py [--figures-dir figures_combined]

Outputs:
    lorekv_architecture.{svg,png}: The method overview, from the TikZ source
        ``assets/src/lorekv_architecture.tex``. Requires ``tectonic`` (or
        ``pdflatex``) and PyMuPDF. The step is skipped if one is missing.
    lorekv_mechanism.gif: An animation of Algorithm 1 on a toy prompt. The tokens,
        queries and scores are synthetic, and the figure shows this.
    paper/*.png: The figures of the paper, made from the PDFs in
        ``--figures-dir`` (``figures_combined``, which is not in the repository).
        The step is skipped if the folder is missing.

Set ``FONT_DIR`` to a directory with the Source Sans 3 TTF files to use the font of
the architecture figure in the GIF.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "assets")

# Palette shared with assets/src/lorekv_architecture.tex.
SURFACE = "#f7f7f4"
CARD = "#ffffff"
RULE = "#e2e1dc"
INK = "#14161a"
INK_2 = "#52514e"
MUTED = "#a3a29c"
BLUE = "#2a78d6"     # LORE-KV / lookahead
BLUE_BG = "#e8f1fc"
ORANGE = "#eb6834"   # prompt window
AQUA = "#1baf7a"     # answer-time queries, evidence

# Paper figure -> asset name (paper numbering in the comment).
PAPER_FIGURES = {
    "montecarlo_queries": "fig1_montecarlo_queries",                    # Fig. 1
    "longbench_mistral_kv_efficiency_pairwise": "fig2_kv_efficiency",  # Fig. 2
    "ruler_main": "fig3_ruler",                                        # Fig. 3
    "accuracy_memory_tradeoff": "fig4_accuracy_memory",                # Fig. 4
    "fig3_cache_distribution": "fig5_output_fidelity",                 # Fig. 5
    "fig6_mismatch_vs_gain": "fig6_gain_vs_mismatch",                  # Fig. 6
    "latency_quality_mistral": "fig7_latency_quality",                 # Fig. 7
    "niah_heatmap_llama_b1024": "fig8_niah",                           # Fig. 8
    "fig2_score_comparison": "fig9_retention_shift",                   # Fig. 9
    "ranking_quality": "fig10_criterion_comparison",                   # Fig. 10
    "fig7_retained_token_composition": "fig11_token_classes",          # Fig. 11
    "fig8_layer_head_analysis": "fig12_layer_head",                    # Fig. 12
    "fig9_budget_equivalence": "fig13_budget_equivalence",             # Fig. 13
    "compute_pareto_mistral": "table13b_compute_pareto",               # Table 13b
    "ruler_long_avg": "ruler_64k_128k",                                # RULER 64K/128K
}


def _setup_fonts():
    """Set the matplotlib font and colors."""
    font_dir = os.environ.get("FONT_DIR")
    family = "DejaVu Sans"
    if font_dir and glob.glob(os.path.join(font_dir, "SourceSans3-*.ttf")):
        for f in glob.glob(os.path.join(font_dir, "SourceSans3-*.ttf")):
            font_manager.fontManager.addfont(f)
        family = "Source Sans 3"
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "mathtext.fontset": "cm",
        "axes.edgecolor": RULE, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "figure.facecolor": SURFACE, "axes.facecolor": CARD,
        "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    })


# ------------------------------------------------------------------ architecture
def architecture():
    """Compile the TikZ source and write the figure as SVG and PNG."""
    tex = os.path.join(OUT, "src", "lorekv_architecture.tex")
    engine = shutil.which("tectonic") or os.environ.get("TECTONIC") or shutil.which("pdflatex")
    try:
        import pymupdf
    except ImportError:
        pymupdf = None
    if not engine or pymupdf is None:
        print("skip architecture: needs tectonic/pdflatex and PyMuPDF")
        return
    with tempfile.TemporaryDirectory() as tmp:
        if os.path.basename(engine).startswith("tectonic"):
            cmd = [engine, "-X", "compile", "--outdir", tmp, tex]
        else:
            cmd = [engine, "-interaction=nonstopmode", "-output-directory", tmp, tex]
        subprocess.run(cmd, check=True, capture_output=True)
        page = pymupdf.open(os.path.join(tmp, "lorekv_architecture.pdf"))[0]
        with open(os.path.join(OUT, "lorekv_architecture.svg"), "w") as f:
            f.write(page.get_svg_image(text_as_path=True))
        page.get_pixmap(dpi=220).save(os.path.join(OUT, "lorekv_architecture.png"))


# --------------------------------------------------------------------- paper figures
def paper_figures(figures_dir):
    """Convert the selected paper PDFs to PNG files in ``assets/paper``."""
    try:
        import pymupdf
    except ImportError:
        print("skip paper figures: needs PyMuPDF")
        return
    if not os.path.isdir(figures_dir):
        print(f"skip paper figures: {figures_dir} not found")
        return
    os.makedirs(os.path.join(OUT, "paper"), exist_ok=True)
    for src, name in PAPER_FIGURES.items():
        path = os.path.join(figures_dir, f"{src}.pdf")
        if not os.path.exists(path):
            print(f"  missing {path}")
            continue
        page = pymupdf.open(path)[0]
        dpi = 420 if page.rect.width < 150 else 320           # narrow panels get more pixels
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        pix.save(os.path.join(OUT, "paper", f"{name}.png"))


# ------------------------------------------------------------------------ mechanism
def _card(fig, rect, title=None):
    """Draw a white card with rounded corners and return an axes inside it."""
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=CARD, ec=RULE, lw=0.8,
                                      zorder=-10))
    if title:
        fig.text(x + 0.015, y + h - 0.045, title, fontsize=10, color=MUTED, weight="semibold")
    ax = fig.add_axes([x + 0.02, y + 0.03, w - 0.04, h - (0.1 if title else 0.06)])
    ax.patch.set_alpha(0)
    return ax


def _ease(p):
    """Return the smoothstep of ``p`` in [0, 1]."""
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def _mix(c0, c1, p):
    """Mix two hex colors."""
    a = np.array(matplotlib.colors.to_rgb(c0))
    b = np.array(matplotlib.colors.to_rgb(c1))
    return tuple(a + (b - a) * p)


def mechanism_gif(frames_per_step=12, hold_frames=15, frame_ms=45):
    """Make the animation of Algorithm 1 on a toy prompt.

    Each of the seven steps has an eased transition of ``frames_per_step`` frames
    and a hold of ``hold_frames`` frames.
    """
    rng = np.random.default_rng(3)
    n, M, B = 40, 4, 12
    window = list(range(n - 4, n))
    evidence = [9, 10, 22, 23]
    prompt_q = rng.normal([-1.6, -0.6], 0.22, (8, 2))
    future_q = rng.normal([1.0, 0.7], 0.32, (10, 2))
    origin = prompt_q.mean(0)
    trajs = [rng.normal([0.9 + 0.25 * np.cos(k), 0.6 + 0.25 * np.sin(k)], 0.3, (6, 2))
             for k in range(M)]
    weights = np.array([0.31, 0.27, 0.24, 0.18])
    s_prompt = rng.uniform(0.05, 0.35, n)
    s_prompt[[3, 4, 15, 30, 31]] = [0.8, 0.7, 0.75, 0.65, 0.7]
    s_look = rng.uniform(0.05, 0.3, n)
    s_look[evidence] = [0.9, 0.85, 0.95, 0.8]
    s_total = s_prompt + s_look
    cand = [i for i in range(1, n) if i not in window]
    top = sorted(cand, key=lambda i: -s_total[i])[: B - 1 - len(window)]
    keep = set([0] + window + top)
    keep_w = set([0] + window + sorted(cand, key=lambda i: -s_prompt[i])[: B - 1 - len(window)])
    thr = min(s_total[i] for i in top) - 0.02
    steps = [
        ("Prefill", "keep the full KV cache and the window queries $Q_W$"),
        ("Monte Carlo lookahead", "sample $M$ short continuations from the frozen model"),
        ("Coverage", "the sampled queries cover the answer-time queries"),
        ("Reliability weights", r"$\omega=\mathrm{softmax}(\beta\bar c-\gamma\bar H-\lambda_D\hat D)$"),
        ("Projected deletion score", r"$S_i=\psi(E_{i,W})+\lambda_L\sum_m\omega_m\psi(E_{i,m})$"),
        ("Select", "keep the top-$B$ tokens of each head"),
        ("Evict and decode", "discard the continuations; decode from the compressed cache"),
    ]
    n_steps = len(steps)

    def draw(step, p):
        e = _ease(p)
        fig = plt.figure(figsize=(9.6, 5.0))
        fig.patch.set_facecolor(SURFACE)
        accent = BLUE if 1 <= step <= 4 else INK_2
        # progress bar
        prog = (step + p) / n_steps
        fig.patches.append(FancyBboxPatch((0.03, 0.965), 0.94, 0.006, boxstyle="square,pad=0",
                                          transform=fig.transFigure, fc=RULE, lw=0))
        fig.patches.append(FancyBboxPatch((0.03, 0.965), 0.94 * prog, 0.006, boxstyle="square,pad=0",
                                          transform=fig.transFigure, fc=BLUE, lw=0))
        fig.text(0.035, 0.895, str(step + 1), fontsize=12, color="white", weight="bold",
                 ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.32", fc=accent, ec=accent))
        fig.text(0.065, 0.895, steps[step][0], fontsize=16, color=INK, weight="bold", va="center")
        fig.text(0.065, 0.84, steps[step][1], fontsize=11.5, color=INK_2, va="center")
        fig.text(0.97, 0.895, "LORE-KV", fontsize=11, color=MUTED, weight="bold", ha="right", va="center")
        fig.text(0.97, 0.035, "schematic, synthetic values", fontsize=8.5, color=MUTED, ha="right")

        # token strip
        ax = _card(fig, (0.03, 0.585, 0.94, 0.2), "prompt KV cache")
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(0, 1.5)
        ax.axis("off")
        for i in range(n):
            base = "#ecebe6"
            win = "#ffd9c9" if (i in window or i == 0) else base
            fc = _mix(base, win, e if step == 0 else 1.0)
            ec = fc
            if step >= 5:
                q = e if step == 5 else 1.0
                if i in keep:
                    target = BLUE if (i not in window and i != 0) else "#a9c9f1"
                    fc = ec = _mix(fc, target, q)
                else:
                    fc = _mix(fc, CARD, q)
                    ec = _mix(fc, RULE, q)
            ax.add_patch(FancyBboxPatch((i - 0.4, 0.15), 0.8, 0.75,
                                        boxstyle="round,pad=0,rounding_size=0.14", fc=fc, ec=ec, lw=1))
            if i in evidence:
                ax.plot(i, 1.12, marker="v", ms=6, color=AQUA)
        ax.text(evidence[0] - 0.8, 1.12, "evidence", fontsize=9, color=AQUA, ha="right", va="center")
        ax.text(n - 2.5, 1.12, "window $W$", fontsize=9, color=ORANGE if step < 5 else INK_2,
                ha="center", va="center")

        # query space
        ax2 = _card(fig, (0.03, 0.08, 0.44, 0.48), "query space (2-D projection)")
        ax2.set_xlim(-2.6, 2.4)
        ax2.set_ylim(-1.5, 1.7)
        ax2.axis("off")
        a0 = e if step == 0 else 1.0
        ax2.scatter(*prompt_q.T, s=34, color=ORANGE, alpha=a0, edgecolor=CARD, lw=0.8, zorder=3,
                    label="window $Q_W$")
        if step >= 2:
            ax2.scatter(*future_q.T, s=95, marker="*", color=AQUA, alpha=e if step == 2 else 1.0,
                        edgecolor=CARD, lw=0.5, zorder=4, label="answer-time queries")
        if step >= 1:
            move = e if step == 1 else 1.0
            for m, t in enumerate(trajs):
                pts = origin + (t - origin) * move
                w_scale = (e if step == 3 else (1.0 if step > 3 else 0.0))
                size = 30 + 140 * weights[m] * w_scale
                ax2.scatter(*pts.T, s=size, color=BLUE, alpha=0.45, edgecolor=CARD, lw=0.6, zorder=2,
                            label="lookahead $Q_{\\mathrm{look}}$" if m == 0 else None)
        ax2.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK_2)

        # right panel
        title = ["decoding has not started", "continuations", "continuations", "trajectory weights",
                 "token scores", "token scores", "outcome"][step]
        ax3 = _card(fig, (0.5, 0.08, 0.47, 0.48), title)
        for s_ in ("top", "right"):
            ax3.spines[s_].set_visible(False)
        if step == 0:
            ax3.axis("off")
            ax3.text(0.5, 0.5, "answer-time queries are\nunknown at eviction time", ha="center",
                     va="center", fontsize=12.5, color=MUTED, transform=ax3.transAxes, alpha=e)
        elif step in (1, 2):
            ax3.axis("off")
            ys = np.linspace(0.82, 0.18, M)
            shown = 6 * M * (e if step == 1 else 1.0)
            for m in range(M):
                ax3.text(0.0, ys[m], f"$y^{{({m + 1})}}$", fontsize=13, color=BLUE, va="center",
                         transform=ax3.transAxes)
                for t in range(6):
                    k = t * M + m
                    if k >= shown:
                        continue
                    ax3.add_patch(FancyBboxPatch((0.12 + t * 0.14, ys[m] - 0.055), 0.115, 0.11,
                                                 boxstyle="round,pad=0,rounding_size=0.025",
                                                 fc=BLUE_BG, ec=BLUE, lw=0.8, transform=ax3.transAxes))
        elif step == 3:
            ax3.barh(np.arange(M)[::-1], weights * e, color=BLUE, height=0.5)
            ax3.set_yticks(np.arange(M)[::-1], [f"$y^{{({m + 1})}}$" for m in range(M)])
            for m, w in enumerate(weights):
                ax3.text(w * e + 0.006, M - 1 - m, f"{w * e:.2f}", va="center", fontsize=10, color=INK_2)
            ax3.set_xlim(0, 0.4)
            ax3.set_xlabel(r"$\omega_m$")
        elif step in (4, 5):
            x = np.arange(n)
            g = e if step == 4 else 1.0
            ax3.bar(x, s_prompt * g, color=ORANGE, width=0.78, label=r"$\psi(E_{i,W})$")
            ax3.bar(x, s_look * g, bottom=s_prompt * g, color=BLUE, width=0.78,
                    label=r"$\lambda_L\sum_m\omega_m\psi(E_{i,m})$")
            if step == 5:
                y = 1.25 + (thr - 1.25) * e
                ax3.axhline(y, color=INK_2, lw=1, ls="--")
                ax3.text(n - 1, y + 0.03, "top-$B$", ha="right", fontsize=9.5, color=INK_2)
            ax3.set_xlim(-1, n)
            ax3.set_ylim(0, 1.3)
            ax3.set_xticks([])
            ax3.legend(loc="upper left", ncol=2, fontsize=9, frameon=False, labelcolor=INK_2)
        else:
            ax3.axis("off")
            rows = [("compressed cache", "$B$ tokens per head", INK),
                    ("continuations", "discarded", INK),
                    ("evidence kept, LORE-KV", f"{len([v for v in evidence if v in keep])} / 4", BLUE),
                    ("evidence kept, window only", f"{len([v for v in evidence if v in keep_w])} / 4", ORANGE)]
            for k, (a, b, col) in enumerate(rows):
                alpha = _ease(e * 1.6 - k * 0.2)
                y = 0.82 - 0.2 * k
                ax3.text(0.02, y, a, fontsize=12, color=INK_2, alpha=alpha, transform=ax3.transAxes)
                ax3.text(0.98, y, b, fontsize=12, ha="right", color=col, weight="bold", alpha=alpha,
                         transform=ax3.transAxes)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    rgb = []
    for step in range(n_steps):
        for k in range(frames_per_step):
            rgb.append(draw(step, (k + 1) / frames_per_step))
        rgb += [rgb[-1]] * (hold_frames - 1)
    # One shared palette keeps colors stable between frames (no flicker).
    sample = rgb[::6]
    sheet = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        sheet.paste(im, (0, k * im.height))
    pal = sheet.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    durations = [frame_ms] * len(frames)
    durations[-1] = 1500
    frames[0].save(os.path.join(OUT, "lorekv_mechanism.gif"), save_all=True,
                   append_images=frames[1:], duration=durations, loop=0, optimize=True,
                   disposal=1)


def main():
    """Make all figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--figures-dir", default=os.path.join(REPO, "figures_combined"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    _setup_fonts()
    architecture()
    paper_figures(a.figures_dir)
    mechanism_gif()
    print("assets:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
