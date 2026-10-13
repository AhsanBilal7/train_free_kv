#!/usr/bin/env python3
"""Draw the README figures into assets/.

    python scripts/make_readme_assets.py

* ``lorekv_pipeline.png``   method overview (Algorithm 1)
* ``lorekv_mechanism.gif``  animated walk-through on a toy prompt (schematic)
* ``results_budget.png``    LongBench and RULER-16K averages vs. cache budget
                            (Mistral-7B-Instruct-v0.2; paper Table 1 and Table 7)
* ``ablation_lookahead.png`` LongBench average vs. number of trajectories M at
                            B = 128 (paper Table 14)

The result charts plot numbers copied from the paper's tables. The animation is
illustrative: its tokens, queries and scores are synthetic.
"""
from __future__ import annotations

import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
BLUE = "#2a78d6"     # LORE-KV
ORANGE = "#eb6834"   # AnDPro / prompt window
AQUA = "#1baf7a"     # true future queries
VIOLET = "#6250d6"   # CriticalKV
PALE_BLUE = "#d6e6f8"
PALE_ORANGE = "#fbe0d4"
PALE_GRAY = "#efeee9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=color, weight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK_2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.3, color=color, shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------- pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    stages = [
        ("Prefill", "full cache K, V\nprompt-window\nqueries $Q_W$", PALE_GRAY),
        ("Monte Carlo\nlookahead", "sample $M$ continuations\nof $T$ tokens, keep their\nqueries $Q^{(m)}_{look}$",
         PALE_BLUE),
        ("Reliability\nweights", r"$R_m=\beta\bar c_m-\gamma\bar H_m-\lambda_D\hat D_m$"
         "\n" r"$\omega=\mathrm{softmax}(R)$", PALE_BLUE),
        ("Projected deletion\nscore", r"$e_{q,i}=\|W_O\,\Delta o_{q,i}\|$" "\n"
         r"$S_i=\psi(E_{i,W})+\lambda_L\sum_m\omega_m\psi(E_{i,m})$", PALE_BLUE),
        ("Evict and\ndecode", "keep top-$B$ tokens per\nhead; discard the\ncontinuations", PALE_GRAY),
    ]
    w, gap, x = 2.15, 0.27, 0.15
    for i, (title, body, fc) in enumerate(stages):
        box(ax, x, 1.95, w, 0.95, title, fc, fs=10.5, weight="bold")
        ax.text(x + w / 2, 1.55, body, ha="center", va="top", fontsize=8.6, color=INK_2,
                linespacing=1.45)
        if i < len(stages) - 1:
            arrow(ax, x + w + 0.03, 2.42, x + w + gap - 0.03, 2.42)
        x += w + gap
    ax.text(0.15, 3.35, "LORE-KV: score the prompt cache with the queries the model is likely to "
            "issue while answering", fontsize=11.5, color=INK, weight="bold")
    ax.text(0.15, 0.12, "Training-free. One-time overhead of $M\\times T$ decoded tokens before "
            "generation; decoding memory and per-token cost equal those of any method at budget $B$.",
            fontsize=8.8, color=INK_2)
    fig.savefig(os.path.join(OUT, "lorekv_pipeline.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------- mechanism
def mechanism_gif():
    rng = np.random.default_rng(3)
    n, M, B = 40, 4, 12
    window = list(range(n - 4, n))
    evidence = [9, 10, 22, 23]                        # tokens the answer will need
    # Toy query geometry: the prompt window sits apart from the answer-time queries.
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

    captions = [
        "1  Prefill: the prompt's KV cache and its last-window queries $Q_W$",
        "2  Sample $M$ short continuations from the frozen model",
        "3  Their queries cover the answer-time queries; $Q_W$ does not",
        "4  Weight each trajectory by its reliability $\\omega_m$",
        "5  Score every token by its projected deletion effect under all queries",
        "6  Keep the top-$B$ tokens; the evidence tokens survive",
        "7  Discard the continuations and decode from the compressed cache",
    ]

    def frame(step):
        fig = plt.figure(figsize=(9.6, 4.6))
        fig.text(0.03, 0.93, captions[step], fontsize=12.5, color=INK, weight="bold")
        fig.text(0.03, 0.03, "Schematic on a toy prompt; tokens, queries and scores are synthetic.",
                 fontsize=8, color=MUTED)
        # --- token strip
        ax = fig.add_axes([0.03, 0.62, 0.94, 0.2])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.6)
        ax.axis("off")
        for i in range(n):
            fc, ec = PALE_GRAY, PALE_GRAY
            if i in window or i == 0:
                fc = "#d8d7d0"
            if step >= 5:
                if i in keep_lorekv:
                    fc = BLUE if i not in window and i != 0 else "#9fc3ee"
                else:
                    fc, ec = SURFACE, GRID
            ax.add_patch(FancyBboxPatch((i - 0.42, 0.25), 0.84, 0.75,
                                        boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1))
            if i in evidence:
                ax.plot(i, 1.22, marker="v", ms=6, color=AQUA)
        ax.text(-0.5, 1.45, "prompt tokens (KV cache)", fontsize=9, color=INK_2)
        ax.text(n - 4.4, 1.45, "window $W$", fontsize=9, color=INK_2)
        ax.text(evidence[0] - 0.4, 1.45, "evidence", fontsize=9, color=AQUA)
        # --- left: query space
        ax2 = fig.add_axes([0.05, 0.1, 0.4, 0.46])
        ax2.set_xlim(-2.6, 2.4)
        ax2.set_ylim(-1.5, 1.9)
        ax2.set_xticks([]), ax2.set_yticks([])
        for s_ in ax2.spines.values():
            s_.set_color(GRID)
        ax2.text(-2.5, 1.65, "query space (2-D projection)", fontsize=9, color=INK_2)
        ax2.scatter(*prompt_q.T, s=28, color=ORANGE, label="prompt window $Q_W$", zorder=3)
        if step >= 2:
            ax2.scatter(*future_q.T, s=60, marker="*", color=AQUA, label="answer-time queries", zorder=4)
        if step >= 1:
            for m, t in enumerate(trajs):
                size = 26 if step < 3 else 26 + 120 * weights[m]
                ax2.scatter(*t.T, s=size, color=BLUE, alpha=0.35 + 0.15 * (m == 0),
                            edgecolor=SURFACE, lw=0.6, label="lookahead $Q_{look}$" if m == 0 else None,
                            zorder=2)
        ax2.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_2)
        # --- right: weights / scores / decoding
        ax3 = fig.add_axes([0.53, 0.1, 0.43, 0.46])
        for s_ in ("top", "right"):
            ax3.spines[s_].set_visible(False)
        if step <= 2:
            ax3.axis("off")
            ys = np.linspace(0.85, 0.2, M)
            for m in range(min(M, 4 if step >= 1 else 0)):
                ax3.text(0.0, ys[m], f"$y^{{({m + 1})}}$", fontsize=11, color=BLUE, va="center")
                for t in range(6):
                    ax3.add_patch(FancyBboxPatch((0.12 + t * 0.13, ys[m] - 0.05), 0.11, 0.1,
                                                 boxstyle="round,pad=0,rounding_size=0.02",
                                                 fc=PALE_BLUE, ec=BLUE, lw=0.8,
                                                 transform=ax3.transAxes))
            if step >= 1:
                ax3.text(0.0, 1.0, "$M=4$ trajectories, $T=6$ tokens each", fontsize=9, color=INK_2)
        elif step == 3:
            ax3.barh(np.arange(M)[::-1], weights, color=BLUE, height=0.55)
            ax3.set_yticks(np.arange(M)[::-1], [f"traj. {m + 1}" for m in range(M)])
            ax3.set_xlabel(r"weight $\omega_m$")
            ax3.set_xlim(0, 0.4)
        elif step in (4, 5):
            x = np.arange(n)
            ax3.bar(x, s_prompt, color=ORANGE, width=0.8, label=r"$\psi(E_{i,W})$ prompt window")
            ax3.bar(x, s_look, bottom=s_prompt, color=BLUE, width=0.8,
                    label=r"$\lambda_L\sum_m\omega_m\psi(E_{i,m})$ lookahead")
            if step == 5:
                thr = sorted([s_total[i] for i in keep_lorekv if i not in window and i != 0])[0]
                ax3.axhline(thr - 0.02, color=INK_2, lw=1, ls="--")
                ax3.text(n - 1, thr + 0.03, "top-$B$ cut", ha="right", fontsize=8.5, color=INK_2)
            ax3.set_xlim(-1, n)
            ax3.set_xticks([])
            ax3.set_ylabel("token score $S_i$")
            ax3.set_ylim(0, 1.25)
            ax3.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False,
                       labelcolor=INK_2)
        else:
            ax3.axis("off")
            missed = [e for e in evidence if e not in keep_prompt]
            lines = [("Compressed cache: $B$ tokens per head", INK),
                     ("Continuations discarded; no extra decode memory", INK_2),
                     (f"Evidence kept: LORE-KV {len([e for e in evidence if e in keep_lorekv])}/4, "
                      f"prompt window only {4 - len(missed)}/4", BLUE)]
            for k, (txt, col) in enumerate(lines):
                ax3.text(0.0, 0.8 - 0.2 * k, txt, fontsize=11, color=col, transform=ax3.transAxes)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)

    frames = [frame(k) for k in range(len(captions))]
    frames[0].save(os.path.join(OUT, "lorekv_mechanism.gif"), save_all=True,
                   append_images=frames[1:], duration=[1900] * (len(frames) - 1) + [3200],
                   loop=0, optimize=True)


# ---------------------------------------------------------------------------- results
def results_budget():
    budgets = [128, 256, 512, 1024]
    longbench = {  # Table 1, Mistral-7B-Instruct-v0.2, 16-task average
        "CriticalKV": [38.42, 40.03, 40.98, 41.83],
        "AnDPro": [38.96, 40.84, 41.62, 42.03],
        "LORE-KV /E": [40.28, 41.38, 41.86, 42.20],
    }
    ruler16k = {  # Table 7, Mistral-7B-Instruct-v0.2, RULER 16K 13-task average
        "CriticalKV": [36.08, 51.95, 63.65, 80.25],
        "AnDPro": [45.20, 57.91, 67.12, 82.06],
        "LORE-KV /E": [51.05, 59.94, 68.28, 82.75],
    }
    colors = {"CriticalKV": VIOLET, "AnDPro": ORANGE, "LORE-KV /E": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, data, title, full in ((axes[0], longbench, "LongBench (16 tasks)", 42.51),
                                  (axes[1], ruler16k, "RULER 16K (13 tasks)", None)):
        for name, ys in data.items():
            ax.plot(budgets, ys, color=colors[name], lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
            ax.annotate(name, (budgets[0], ys[0]), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8.5, color=INK_2)
        if full:
            ax.axhline(full, color=MUTED, lw=1, ls="--")
            ax.text(budgets[-1], full + 0.15, "full cache", ha="right", va="bottom", fontsize=8.5,
                    color=INK_2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlim(70, 1300)
        ax.set_xlabel("cache budget $B$ (tokens per head)")
        ax.set_ylabel("average score")
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.grid(axis="y", color=GRID, lw=0.8)
    fig.text(0.01, -0.02, "Mistral-7B-Instruct-v0.2. Numbers from the paper's Tables 1 and 7.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "results_budget.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def ablation_lookahead():
    Ms = [0, 1, 2, 4, 8]
    scores = [35.87, 39.81, 40.17, 40.28, 40.69]  # Table 14, B = 128
    x = np.arange(len(Ms))
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.axhline(38.96, color=ORANGE, lw=1.5, ls="--", zorder=2)
    ax.text(-0.15, 39.05, "AnDPro 38.96", ha="left", va="bottom", fontsize=9, color=INK_2)
    ax.plot(x, scores, color=BLUE, lw=2, zorder=3)
    ax.scatter(x[1:], scores[1:], s=60, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4)
    ax.scatter(x[:1], scores[:1], s=60, color=MUTED, edgecolor=SURFACE, lw=1.5, zorder=4)
    for k, v in enumerate(scores):
        ax.text(k, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, ["0\n(prompt window)"] + [str(m) for m in Ms[1:]])
    ax.set_xlabel("lookahead trajectories $M$ ($T=8$)")
    ax.set_ylabel("LongBench average")
    ax.set_ylim(35, 41.6)
    ax.set_xlim(-0.4, len(Ms) - 0.6)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title("One sampled future recovers most of the gain", loc="left", fontsize=11,
                 weight="bold")
    fig.text(0.01, -0.03, "Mistral-7B-Instruct-v0.2, $B=128$. Paper Table 14.", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_lookahead.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline()
    results_budget()
    ablation_lookahead()
    mechanism_gif()
    print("wrote", sorted(os.listdir(OUT)))
