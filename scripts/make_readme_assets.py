#!/usr/bin/env python3
"""🎨 Build the README and project-page figures into ``assets/``.

    python scripts/make_readme_assets.py [--figures-dir figures_combined]

Outputs

* ``lorekv_architecture.{svg,png}`` -- the method overview, drawn in TikZ from
  ``assets/src/lorekv_architecture.tex``. Needs ``tectonic`` (or ``pdflatex``) and
  PyMuPDF; skipped with a message if either is missing.
* ``lorekv_mechanism.gif`` -- an animated walk-through of Algorithm 1 on a toy
  prompt. Its tokens, queries and scores are synthetic and labelled as such.
* ``paper/*.png`` -- figures of the paper, rasterized from the PDFs in
  ``--figures-dir`` (the paper's ``figures_combined`` folder, which is not part of
  the repository). Skipped if the folder is absent.

Set ``FONT_DIR`` to a directory with Source Sans 3 TTF files to typeset the GIF in
the same face as the architecture figure; otherwise matplotlib's default is used.
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

# Palette shared with assets/src/lorekv_architecture.tex and website/.
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
    """Compile the TikZ source and export it as SVG and PNG."""
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
    """A white rounded card on the page surface; returns an axes inside it."""
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=CARD, ec=RULE, lw=0.8,
                                      zorder=-10))
    if title:
        fig.text(x + 0.015, y + h - 0.045, title, fontsize=10, color=MUTED, weight="semibold")
    ax = fig.add_axes([x + 0.02, y + 0.03, w - 0.04, h - (0.1 if title else 0.06)])
    ax.patch.set_alpha(0)
    return ax


def mechanism_gif():
    rng = np.random.default_rng(3)
    n, M, B = 40, 4, 12
    window = list(range(n - 4, n))
    evidence = [9, 10, 22, 23]                       # tokens the answer will need
    prompt_q = rng.normal([-1.6, -0.6], 0.25, (8, 2))
    future_q = rng.normal([1.0, 0.7], 0.35, (10, 2))
    trajs = [rng.normal([0.9 + 0.25 * np.cos(k), 0.6 + 0.25 * np.sin(k)], 0.33, (6, 2))
             for k in range(M)]
    weights = np.array([0.31, 0.27, 0.24, 0.18])
    s_prompt = rng.uniform(0.05, 0.35, n)
    s_prompt[[3, 4, 15, 30, 31]] = [0.8, 0.7, 0.75, 0.65, 0.7]
    s_look = rng.uniform(0.05, 0.3, n)
    s_look[evidence] = [0.9, 0.85, 0.95, 0.8]
    s_total = s_prompt + s_look

    def keep_set(score):
        cand = [i for i in range(1, n) if i not in window]
        top = sorted(cand, key=lambda i: -score[i])[: B - 1 - len(window)]
        return set([0] + window + top)

    keep_prompt, keep_lorekv = keep_set(s_prompt), keep_set(s_total)
    steps = [
        ("Prefill", "keep the full KV cache and the last-window queries $Q_W$"),
        ("Monte Carlo lookahead", "sample $M$ short continuations from the frozen model"),
        ("Coverage", "their queries land where the answer-time queries live; $Q_W$ does not"),
        ("Reliability weights", "weight each trajectory by $\\omega_m=\\mathrm{softmax}(R)_m$"),
        ("Projected deletion score", "score each token under the window and all trajectories"),
        ("Select", "keep the top-$B$ tokens per head; the evidence survives"),
        ("Evict and decode", "discard the continuations and decode from the compressed cache"),
    ]

    def frame(step):
        fig = plt.figure(figsize=(9.6, 5.0))
        fig.patch.set_facecolor(SURFACE)
        # header: numbered badge, title, subtitle, progress dots
        badge = BLUE if 1 <= step <= 4 else INK_2
        fig.text(0.035, 0.915, str(step + 1), fontsize=12, color="white", weight="bold",
                 ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.32", fc=badge, ec=badge))
        fig.text(0.065, 0.915, steps[step][0], fontsize=16, color=INK, weight="bold", va="center")
        fig.text(0.065, 0.858, steps[step][1], fontsize=11.5, color=INK_2, va="center")
        for k in range(len(steps)):
            fig.patches.append(FancyBboxPatch((0.78 + k * 0.03, 0.905), 0.02, 0.02,
                                              boxstyle="round,pad=0,rounding_size=0.01",
                                              transform=fig.transFigure, lw=0,
                                              fc=badge if k == step else RULE))
        fig.text(0.97, 0.035, "schematic: toy prompt, synthetic values", fontsize=8.5,
                 color=MUTED, ha="right")
        fig.text(0.03, 0.035, "LORE-KV", fontsize=9, color=MUTED, weight="bold")

        # token strip card
        ax = _card(fig, (0.03, 0.6, 0.94, 0.2), "prompt KV cache")
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(0, 1.5)
        ax.axis("off")
        for i in range(n):
            fc, ec = "#ecebe6", "#ecebe6"
            if i in window or i == 0:
                fc = ec = "#ffd9c9" if step < 5 else fc
            if step >= 5:
                if i in keep_lorekv:
                    fc = ec = BLUE if i not in window and i != 0 else "#a9c9f1"
                else:
                    fc, ec = CARD, RULE
            ax.add_patch(FancyBboxPatch((i - 0.4, 0.15), 0.8, 0.75,
                                        boxstyle="round,pad=0,rounding_size=0.14", fc=fc, ec=ec, lw=1))
            if i in evidence:
                ax.plot(i, 1.12, marker="v", ms=6, color=AQUA)
        ax.text(evidence[0] - 0.8, 1.12, "evidence", fontsize=9, color=AQUA, ha="right", va="center")
        ax.text(n - 4.4, 1.32, "window $W$", fontsize=9, color=ORANGE if step < 5 else INK_2)

        # query-space card
        ax2 = _card(fig, (0.03, 0.08, 0.44, 0.48), "query space (2-D projection)")
        ax2.set_xlim(-2.6, 2.4)
        ax2.set_ylim(-1.5, 1.7)
        ax2.axis("off")
        ax2.scatter(*prompt_q.T, s=34, color=ORANGE, label="prompt window $Q_W$", zorder=3,
                    edgecolor=CARD, lw=0.8)
        if step >= 2:
            ax2.scatter(*future_q.T, s=95, marker="*", color=AQUA, label="answer-time queries",
                        zorder=4, edgecolor=CARD, lw=0.5)
        if step >= 1:
            for m, t in enumerate(trajs):
                size = 30 if step < 3 else 30 + 140 * weights[m]
                ax2.scatter(*t.T, s=size, color=BLUE, alpha=0.4 + 0.12 * (m == 0), edgecolor=CARD,
                            lw=0.6, label="lookahead $Q_{\\mathrm{look}}$" if m == 0 else None, zorder=2)
        ax2.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK_2)

        # right card: trajectories / weights / scores / outcome
        title = {0: "decoding has not started", 1: "trajectories", 2: "trajectories",
                 3: "reliability weights", 4: "token scores $S_i$", 5: "token scores $S_i$",
                 6: "outcome"}[step]
        ax3 = _card(fig, (0.5, 0.08, 0.47, 0.48), title)
        for s_ in ("top", "right"):
            ax3.spines[s_].set_visible(False)
        if step <= 2:
            ax3.axis("off")
            if step == 0:
                ax3.text(0.5, 0.5, "the future queries are unknown\nat eviction time", ha="center",
                         va="center", fontsize=12, color=MUTED, transform=ax3.transAxes)
            ys = np.linspace(0.82, 0.18, M)
            for m in range(M if step >= 1 else 0):
                ax3.text(0.0, ys[m], f"$y^{{({m + 1})}}$", fontsize=13, color=BLUE, va="center",
                         transform=ax3.transAxes)
                for t in range(6):
                    ax3.add_patch(FancyBboxPatch((0.12 + t * 0.14, ys[m] - 0.055), 0.115, 0.11,
                                                 boxstyle="round,pad=0,rounding_size=0.025",
                                                 fc=BLUE_BG, ec=BLUE, lw=0.8, transform=ax3.transAxes))
        elif step == 3:
            ax3.barh(np.arange(M)[::-1], weights, color=BLUE, height=0.5)
            ax3.set_yticks(np.arange(M)[::-1], [f"$y^{{({m + 1})}}$" for m in range(M)])
            for m, w in enumerate(weights):
                ax3.text(w + 0.006, M - 1 - m, f"{w:.2f}", va="center", fontsize=10, color=INK_2)
            ax3.set_xlabel(r"$\omega_m$")
            ax3.set_xlim(0, 0.4)
        elif step in (4, 5):
            x = np.arange(n)
            ax3.bar(x, s_prompt, color=ORANGE, width=0.78, label=r"$\psi(E_{i,W})$")
            ax3.bar(x, s_look, bottom=s_prompt, color=BLUE, width=0.78,
                    label=r"$\lambda_L\sum_m\omega_m\psi(E_{i,m})$")
            if step == 5:
                thr = min(s_total[i] for i in keep_lorekv if i not in window and i != 0)
                ax3.axhline(thr - 0.02, color=INK_2, lw=1, ls="--")
                ax3.text(n - 1, thr + 0.03, "top-$B$", ha="right", fontsize=9.5, color=INK_2)
            ax3.set_xlim(-1, n)
            ax3.set_ylim(0, 1.3)
            ax3.set_xticks([])
            ax3.legend(loc="upper left", ncol=2, fontsize=9, frameon=False, labelcolor=INK_2)
        else:
            ax3.axis("off")
            kept = len([e for e in evidence if e in keep_lorekv])
            kept_w = len([e for e in evidence if e in keep_prompt])
            rows = [("compressed cache", "$B$ tokens per head"),
                    ("continuations", "discarded"),
                    ("evidence kept, LORE-KV", f"{kept} / 4"),
                    ("evidence kept, prompt window only", f"{kept_w} / 4")]
            for k, (a, b) in enumerate(rows):
                y = 0.82 - 0.2 * k
                ax3.text(0.02, y, a, fontsize=12, color=INK_2, transform=ax3.transAxes)
                ax3.text(0.98, y, b, fontsize=12, ha="right", transform=ax3.transAxes,
                         color=BLUE if k == 2 else (ORANGE if k == 3 else INK), weight="bold")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=160)

    frames = [frame(k) for k in range(len(steps))]
    frames[0].save(os.path.join(OUT, "lorekv_mechanism.gif"), save_all=True,
                   append_images=frames[1:], duration=[2200] * (len(frames) - 1) + [3600],
                   loop=0, optimize=True)


def main():
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
