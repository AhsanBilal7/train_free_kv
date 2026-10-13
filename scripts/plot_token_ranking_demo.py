"""Draw the figures of the token-ranking comparison (paper Figures 9-11).

The script reads the output of ``scripts/demo_token_ranking.py`` in
``demos/<run>/`` and draws:

ranking_quality.{png,pdf}:
    (a) Recall@B against the oracle, for each budget.
    (b) Rank correlation with the oracle, for each layer.
    (c) Attention-output fidelity of the kept cache.

ranking_evidence.{png,pdf}:
    (a) Retention and enrichment of the answer tokens.
    (b) The kept positions of one prompt, with the answer positions marked.

Usage::

    python scripts/plot_token_ranking_demo.py                     # newest run in demos/
    python scripts/plot_token_ranking_demo.py --run demos/<run>
    python scripts/plot_token_ranking_demo.py --run demos/<run> --out-dir figures
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
GREY_DARK, GREY_MID, INK = "#4D4D4D", "#767676", "#272727"

# Semantic mapping: baselines warm/neutral, the proposed method blue.
METHOD_STYLE = {
    "criticalkv":    ("CriticalKV",        PALETTE["red_strong"], "o", "--"),
    "andpro":        ("AnDPro",            PALETTE["violet"],     "s", "--"),
    "lorekv_window": (r"Window-only ($\lambda_L{=}0$)", PALETTE["teal"], "^", ":"),
    "lorekv":        ("LORE-KV",           PALETTE["blue_main"],  "*", "-"),
    "oracle":        ("Oracle (future queries)", GREY_MID,        "D", "-."),
}
ORDER = ["criticalkv", "andpro", "lorekv_window", "lorekv"]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 15
    axes_linewidth: float = 2.0
    use_tex: bool = False
    font_family: tuple = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    style = style or FigureStyle()
    matplotlib.rcParams.update({
        "font.family": list(style.font_family),
        "font.size": style.font_size,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "mathtext.fontset": "dejavusans",
        "text.usetex": style.use_tex, "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return style


def create_subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(np.asarray(axes)).ravel()


_VALID = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kw):
    out_path = Path(out_path)
    formats = formats or (["pdf", "png"] if not out_path.suffix else [out_path.suffix[1:]])
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


def make_grouped_bar(ax, categories, series, labels, colors, ylabel=None, annotate=False,
                     yerr=None, hatches=None, width=None):
    """Draw grouped bars with black edges and optional value labels."""
    n_g = len(series)
    x = np.arange(len(categories), dtype=float)
    width = width or 0.8 / max(1, n_g)
    for i, (vals, lab) in enumerate(zip(series, labels)):
        off = (i - (n_g - 1) / 2) * width
        err = None if yerr is None else np.asarray(yerr[i])
        bars = ax.bar(x + off, vals, width * 0.92, label=lab, color=colors[i],
                      edgecolor="black", linewidth=1.1,
                      hatch=None if hatches is None else hatches[i],
                      yerr=err, capsize=3, error_kw=dict(lw=1.2, ecolor=GREY_DARK))
        if annotate:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9, color=GREY_DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def figure_legend(fig, axes, ncol=4, y=1.02):
    """Draw one legend without a frame for the full figure."""
    handles, labels, seen = [], [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, ncol=ncol, loc="upper center",
                   bbox_to_anchor=(0.5, y), fontsize=13, frameon=False)
    return handles, labels


def make_trend(ax, x, y_series, labels, colors, markers=None, styles=None,
               ylabel=None, xlabel=None, yerr=None):
    for i, (y, lab) in enumerate(zip(y_series, labels)):
        ax.plot(x, y, label=lab, color=colors[i], lw=2.2,
                marker=None if markers is None else markers[i], ms=8,
                ls="-" if styles is None else styles[i])
        if yerr is not None and yerr[i] is not None:
            y, e = np.asarray(y), np.asarray(yerr[i])
            ax.fill_between(x, y - e, y + e, color=colors[i], alpha=0.15, lw=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


# ---------------------------------------------------------------------------
# Loading a demo run
# ---------------------------------------------------------------------------
def newest_run(demos_dir):
    runs = [d for d in glob.glob(os.path.join(demos_dir, "*"))
            if os.path.isfile(os.path.join(d, "summary_long.csv"))]
    if not runs:
        raise SystemExit(f"no demo runs with summary_long.csv under {demos_dir}; "
                         f"run scripts/demo_token_ranking.py first")
    return max(runs, key=os.path.getmtime)


def load_rows(run_dirs):
    rows = []
    for run in run_dirs:
        with open(os.path.join(run, "summary_long.csv")) as f:
            for r in csv.DictReader(f):
                r["budget"] = int(r["budget"])
                r["value"] = float(r["value"])
                r["benchmark"] = "RULER" if r["dataset"].startswith("ruler") else "LongBench"
                rows.append(r)
    return rows


def recall_metric(rows):
    """Return ``recall_cand`` (protected positions excluded) if the run has it, else ``recall``."""
    return "recall_cand" if any(r["metric"] == "recall_cand" for r in rows) else "recall"


def stat(rows, benchmark=None, budget=None, method=None, metric=None):
    """Return the mean and standard error of the selected rows, or (nan, nan)."""
    v = [r["value"] for r in rows
         if (benchmark is None or r["benchmark"] == benchmark)
         and (budget is None or r["budget"] == budget)
         and (method is None or r["method"] == method)
         and (metric is None or r["metric"] == metric)]
    if not v:
        return float("nan"), float("nan")
    a = np.asarray(v)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def run_benchmark(run):
    """Return the benchmark of a run folder ('LongBench' or 'RULER')."""
    for r in load_rows([run]):
        return r["benchmark"]
    return "?"


def load_per_layer(run_dirs, key):
    """Stack a per-layer array (for example 'lorekv__spearman') over all sample files."""
    out = []
    for run in run_dirs:
        for f in sorted(glob.glob(os.path.join(run, "samples", "*.npz"))):
            d = np.load(f, allow_pickle=True)
            if key in d:
                out.append(np.asarray(d[key], dtype=np.float64))
    if not out:
        return None
    L = min(len(a) for a in out)
    return np.vstack([a[:L] for a in out])


def pick_profile(run_dirs, prefer=None):
    cands = []
    for run in run_dirs:
        cands += sorted(glob.glob(os.path.join(run, "profiles", "*.npz")))
    if not cands:
        return None
    if prefer:
        for c in cands:
            if prefer in os.path.basename(c):
                return np.load(c, allow_pickle=True)
    return np.load(cands[0], allow_pickle=True)


# ---------------------------------------------------------------------------
# Figure 1: ranking quality
# ---------------------------------------------------------------------------
def _cats(benches, budgets):
    """Return the bar-group labels. Add the benchmark name only if a panel has more than one benchmark."""
    if len(benches) == 1:
        return [f"$B={bd}$" for bd in budgets]
    return [f"{b}\nB={bd}" for b in benches for bd in budgets]


def _panel_metric_vs_budget(ax, rows, bench, budgets, metric, ylabel, tag,
                            invert=False, logy=False):
    """Draw one line for each criterion over the budgets.

    With ``invert`` the panel shows 1 - value. A fidelity of 0.82-0.96 shows small
    differences more clearly as an error 1 - cos.
    """
    for meth in ORDER:
        lab, col, mk, st = METHOD_STYLE[meth]
        mu = np.array([stat(rows, bench, b, meth, metric)[0] for b in budgets], dtype=float)
        se = np.array([stat(rows, bench, b, meth, metric)[1] for b in budgets], dtype=float)
        if invert:
            mu = 1.0 - mu
        ax.plot(budgets, mu, color=col, lw=2.2, marker=mk, ms=9, ls=st, label=lab)
        ax.fill_between(budgets, mu - se, mu + se, color=col, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("cache budget $B$ (tokens)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(tag, fontsize=15, loc="left")


def _panel_recall(ax, rows, benches, budgets, rec_key, tag):
    cats = _cats(benches, budgets)
    series, errs, labels, colors = [], [], [], []
    for meth in ORDER:
        lab, col, _, _ = METHOD_STYLE[meth]
        series.append([stat(rows, b, bd, meth, rec_key)[0] for b in benches for bd in budgets])
        errs.append([stat(rows, b, bd, meth, rec_key)[1] for b in benches for bd in budgets])
        labels.append(lab); colors.append(col)
    # In the per-benchmark row layout the y-label is the benchmark name and the
    # quantity is defined in the figure caption, so the axis stays clean.
    ylab = None if len(benches) == 1 else ("Recall@$B$ vs. oracle ranking"
                                           + (" (candidates)" if rec_key == "recall_cand" else ""))
    make_grouped_bar(ax, cats, series, labels, colors, yerr=errs, ylabel=ylab,
                     annotate=False, hatches=[None, "//", "..", None])
    finite = np.concatenate([np.asarray(x)[np.isfinite(x)] for x in series])
    if finite.size:
        ax.set_ylim(max(0.0, finite.min() - 0.06), min(1.0, finite.max() + 0.06))
    ax.set_title(f"{tag}  Agreement with the future-query oracle", fontsize=15, loc="left")


def _panel_spearman(ax, run_dirs, tag):
    xs, ys, labs, cols, sts = None, [], [], [], []
    for meth in ORDER:
        arr = load_per_layer(run_dirs, f"{meth}__spearman")
        if arr is None:
            continue
        lab, col, _, st = METHOD_STYLE[meth]
        xs = np.arange(arr.shape[1])
        ys.append(np.nanmean(arr, axis=0)); labs.append(lab); cols.append(col); sts.append(st)
    if ys:
        make_trend(ax, xs, ys, labs, cols, markers=[None] * len(ys), styles=sts,
                   xlabel="Layer", ylabel=r"Spearman $\rho$ with oracle score")
    ax.set_title(f"{tag}  Rank agreement across depth", fontsize=15, loc="left")


def _panel_fidelity(ax, rows, benches, budgets, tag):
    cats = _cats(benches, budgets)
    series, errs, labels, colors = [], [], [], []
    for meth in ORDER:
        lab, col, _, _ = METHOD_STYLE[meth]
        series.append([stat(rows, b, bd, meth, "fidelity")[0] for b in benches for bd in budgets])
        errs.append([stat(rows, b, bd, meth, "fidelity")[1] for b in benches for bd in budgets])
        labels.append(lab); colors.append(col)
    make_grouped_bar(ax, cats, series, labels, colors, yerr=errs,
                     ylabel=r"$\cos(\mathbf{o}_{\mathcal{R}},\mathbf{o}_{\mathrm{full}})$",
                     hatches=[None, "//", "..", None])
    finite = np.concatenate([np.asarray(x)[np.isfinite(x)] for x in series])
    if finite.size:
        ax.set_ylim(max(0.0, finite.min() - 0.04), min(1.0, finite.max() + 0.02))
    ax.set_title(f"{tag}  Attention-output fidelity on future queries", fontsize=15, loc="left")


def figure_quality(rows, run_dirs, out_base, layout="combined"):
    """Draw the ranking-quality figure.

    ``layout="combined"`` gives 1 x 3 panels with both benchmarks in each panel.
    ``layout="rows"`` gives 2 x 3 panels with one benchmark in each row.
    """
    budgets = sorted({r["budget"] for r in rows})
    benches = [b for b in ("LongBench", "RULER") if any(r["benchmark"] == b for r in rows)]
    rec_key = recall_metric(rows)

    if layout == "cols" and len(benches) > 1:
        # columns = benchmarks, rows = panel types (the matlab_figures convention):
        # a column header names the benchmark, the row identity comes from the
        # y-label, and the panel letter sits inside the axes.
        fig, axes = create_subplots(3, 2, figsize=(13, 12.0))
        for bi, bench in enumerate(benches):
            sub = [r for r in rows if r["benchmark"] == bench]
            runs_b = [d for d in run_dirs if run_benchmark(d) == bench]
            _panel_metric_vs_budget(axes[0 + bi], sub, bench, budgets, rec_key,
                                    "Recall@$B$ vs. oracle\n(candidates)", "")
            _panel_metric_vs_budget(axes[2 + bi], sub, bench, budgets, "fidelity",
                                    r"$1-\cos(\mathbf{o}_{\mathcal{R}},\mathbf{o}_{\mathrm{full}})$"
                                    "\n(output error, lower is better)", "", invert=True,
                                    logy=True)
            _panel_spearman(axes[4 + bi], runs_b, "")
            axes[0 + bi].set_xlabel("")             # row 2 below shares the budget axis
            axes[0 + bi].set_xticklabels([])
            for k, letter in enumerate("ace" if bi == 0 else "bdf"):
                ax = axes[2 * k + bi]
                ax.set_title("", loc="left")
                corner = (0.03, 0.96, "top") if k == 1 else (0.03, 0.04, "bottom")
                ax.text(corner[0], corner[1], f"({letter})", transform=ax.transAxes,
                        ha="left", va=corner[2], fontsize=14, color=INK,
                        bbox=dict(fc="white", ec="none", pad=1.5))
            axes[0 + bi].set_title(bench, fontsize=16, pad=10)
        for r in range(3):                      # one y-scale per row, labels on the left
            if r == 1:                          # the error row is log-scaled; matching the
                lo = min(ax.get_ylim()[0] for ax in axes[2:4])   # limits still helps, but the
                hi = max(ax.get_ylim()[1] for ax in axes[2:4])   # ticks must stay logarithmic
                for c, ax in enumerate(axes[2:4]):
                    ax.set_ylim(lo, hi)
                    if c:
                        ax.set_ylabel("")
                        ax.set_yticklabels([], minor=True)
                        ax.set_yticklabels([])
                continue
            lo = min(ax.get_ylim()[0] for ax in axes[2 * r: 2 * r + 2])
            hi = max(ax.get_ylim()[1] for ax in axes[2 * r: 2 * r + 2])
            for c, ax in enumerate(axes[2 * r: 2 * r + 2]):
                ax.set_ylim(lo, hi)
                if c:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
    elif layout == "rows" and len(benches) > 1:
        # rows = benchmarks, columns = metrics (matches matlab/make_ranking_quality.m):
        # the benchmark names the row through the left-hand y-label, the quantity
        # names each axis, and the panel letter sits inside the axes.
        fig, axes = create_subplots(2, 3, figsize=(18, 9.0))
        for bi, bench in enumerate(benches):
            sub = [r for r in rows if r["benchmark"] == bench]
            runs_b = [d for d in run_dirs if run_benchmark(d) == bench]
            last = bi == len(benches) - 1
            _panel_metric_vs_budget(axes[3 * bi + 0], sub, bench, budgets, rec_key,
                                    f"$\\bf{{{bench}}}$\nRecall@$B$ vs. oracle", "")
            _panel_metric_vs_budget(axes[3 * bi + 1], sub, bench, budgets, "fidelity",
                                    r"$1-\cos(\mathbf{o}_{\mathcal{R}},\mathbf{o}_{\mathrm{full}})$",
                                    "", invert=True, logy=True)
            _panel_spearman(axes[3 * bi + 2], runs_b, "")
            for k, letter in enumerate("abc" if bi == 0 else "def"):
                ax = axes[3 * bi + k]
                ax.set_title("", loc="left")
                corner = (0.03, 0.96, "top") if k != 2 else (0.03, 0.04, "bottom")
                ax.text(corner[0], corner[1], f"({letter})", transform=ax.transAxes,
                        ha="left", va=corner[2], fontsize=14, color=INK,
                        bbox=dict(fc="white", ec="none", pad=1.5))
                if not last:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
        for c in range(3):                  # one scale per metric, both rows
            col = [axes[c], axes[3 + c]]
            lo = min(ax.get_ylim()[0] for ax in col)
            hi = max(ax.get_ylim()[1] for ax in col)
            for ax in col:
                ax.set_ylim(lo, hi)
    else:
        fig, axes = create_subplots(1, 3, figsize=(19, 5.6))
        _panel_recall(axes[0], rows, benches, budgets, rec_key, "(a)")
        _panel_spearman(axes[1], run_dirs, "(b)")
        _panel_fidelity(axes[2], rows, benches, budgets, "(c)")

    fig.tight_layout(pad=1.6)
    figure_legend(fig, axes, ncol=4, y=1.06 if layout != "rows" else 1.03)
    return finalize_figure(fig, out_base, formats=["png", "pdf"], dpi=300, pad=0.06)


# ---------------------------------------------------------------------------
# Figure 2: evidence retention
# ---------------------------------------------------------------------------
def figure_evidence(rows, run_dirs, out_base, prefer_profile=None, bins=400):
    budgets = sorted({r["budget"] for r in rows})
    benches = [b for b in ("LongBench", "RULER") if any(r["benchmark"] == b for r in rows)]
    B0 = budgets[0]
    fig, axes = create_subplots(1, 2, figsize=(19, 5.6),
                               gridspec_kw={"width_ratios": [1.0, 1.45]})

    # (a) answer/evidence-token retention and enrichment
    ax = axes[0]
    cats = [f"{b}\nB={bd}" for b in benches for bd in budgets]
    series, errs, labels, colors = [], [], [], []
    for meth in ORDER:
        lab, col, _, _ = METHOD_STYLE[meth]
        series.append([stat(rows, b, bd, meth, "evidence_enrich")[0]
                       for b in benches for bd in budgets])
        errs.append([stat(rows, b, bd, meth, "evidence_enrich")[1]
                     for b in benches for bd in budgets])
        labels.append(lab); colors.append(col)
    make_grouped_bar(ax, cats, series, labels, colors, yerr=errs, annotate=False,
                     ylabel=r"$P(\mathrm{answer}\mid\mathcal{R})\,/\,P(\mathrm{answer}\mid X)$",
                     hatches=[None, "//", "..", None])
    ax.axhline(1.0, color=GREY_MID, lw=1.4, ls=(0, (4, 3)))
    ax.text(ax.get_xlim()[1], 1.0, " no enrichment", va="center", ha="left",
            fontsize=11, color=GREY_MID)
    ax.set_title("(a)  Answer-token enrichment in the retained cache", fontsize=15, loc="left")

    # (b) retention strip for one prompt
    ax = axes[1]
    prof = pick_profile(run_dirs, prefer_profile)
    if prof is not None:
        n = int(prof["n"]) if "n" in prof else None
        rows_drawn = []
        # the oracle row is the ceiling: what a criterion with the true future queries keeps
        for meth in ORDER + ["oracle"]:
            key = f"keepfrac_b{B0}_{meth}"
            if key in prof:
                rows_drawn.append((meth, np.asarray(prof[key], dtype=np.float64)))
        if rows_drawn:
            mat = np.vstack([r[1] for r in rows_drawn])
            n_tok = mat.shape[1]
            # A single retained token is sub-pixel on a long prompt: bin the positions
            # so the retention pattern is actually visible, and say so on the axis.
            nb = min(bins, n_tok)
            edges = np.linspace(0, n_tok, nb + 1).astype(int)
            # max within the bin, not mean: at B=128 of several thousand tokens a mean
            # washes every bin out to ~0.03 and the pattern disappears.
            binned = np.stack([mat[:, a:max(a + 1, b)].max(axis=1)
                               for a, b in zip(edges[:-1], edges[1:])], axis=1)
            im = ax.imshow(binned, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                           interpolation="nearest",
                           extent=[0, n_tok, len(rows_drawn) - 0.5, -0.5])
            ax.set_yticks(range(len(rows_drawn)))
            ax.set_yticklabels([METHOD_STYLE[m][0] for m, _ in rows_drawn], fontsize=13)
            ax.set_xlabel(f"Prompt token position  ({n_tok} tokens, {nb} bins)")
            cb = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
            cb.set_label("heads retaining (max in bin)", fontsize=12)
            ans = (np.asarray(prof["answer_positions"]).ravel()
                   if "answer_positions" in prof else np.array([]))
            lo, hi = len(rows_drawn) - 0.42, len(rows_drawn) - 0.02
            ax.set_ylim(len(rows_drawn) + 0.05, -0.55)
            if ans.size:
                ax.vlines(ans, lo, hi, color=PALETTE["red_strong"], lw=1.6,
                          label="gold-answer tokens")
            ax.axhline(len(rows_drawn) - 0.5, color=GREY_MID, lw=1.0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    ax.set_title(f"(b)  Which positions survive eviction  ($B={B0}$, mid layer)",
                 fontsize=15, loc="left")

    fig.tight_layout(pad=1.6)
    figure_legend(fig, axes, ncol=5, y=1.06)
    return finalize_figure(fig, out_base, formats=["png", "pdf"], dpi=300, pad=0.06)


def figure_criteria(rows, out_base):
    """Draw a diagram of the input, the measurement and the limit of each criterion.

    The bottom band shows the measured values of the run.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, axes = create_subplots(1, 1, figsize=(16.5, 6.4))
    ax = axes[0]
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.set_axis_off()

    spans = [(1.0, 32.0), (34.5, 65.5), (68.0, 99.0)]
    cols = [
        ("criticalkv", "prompt-window attention mass",
         r"$S_i=\mathrm{pool}\left(\mathrm{mean}_{q\in\mathcal{Q}_\mathcal{W}}\,a_{q,i}\right)$",
         "blind to what the value carries:\ntwo tokens with equal attention\nscore the same"),
        ("andpro", "anchor-direction value projection",
         r"$S_c=\sum_{q\in\mathcal{Q}_\mathcal{W}}\langle\sum_{i\in c}a_{q,i}\mathbf{v}_i,\ "
         r"\sum_{j\in\mathrm{drop}}a_{q,j}\mathbf{v}_j\rangle$",
         "value-aware, but still scored\nonly against prompt-end\nqueries"),
        ("lorekv", "projected deletion under sampled futures",
         r"$S_i=\rho(\mathcal{E}_{i,\mathcal{W}})+\lambda_L\sum_m\omega_m\rho(\mathcal{E}_{i,m})$",
         r"sees the response side: $M$ sampled" "\n" r"continuations weighted by $\omega_m$,"
         "\nscored through $\mathbf{W}^{O}$"),
    ]

    for (x0, x1), (meth, subtitle, formula, note) in zip(spans, cols):
        lab, col, _, _ = METHOD_STYLE[meth]
        is_ours = meth == "lorekv"
        ax.add_patch(FancyBboxPatch((x0, 6.0), x1 - x0, 34.0,
                                    boxstyle="round,pad=0,rounding_size=1.1",
                                    linewidth=2.0 if is_ours else 1.5,
                                    edgecolor=col if is_ours else PALETTE["neutral"],
                                    facecolor="#F7F8FA", zorder=1))
        ax.add_patch(FancyBboxPatch((x0 + 1.4, 35.4), x1 - x0 - 2.8, 3.6,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    linewidth=0, facecolor=col, zorder=2))
        ax.text((x0 + x1) / 2, 37.2, lab, ha="center", va="center", color="white",
                fontsize=15, fontweight="bold", zorder=3)
        ax.text((x0 + x1) / 2, 33.4, subtitle, ha="center", va="center",
                fontsize=12, color=GREY_DARK, zorder=3)

        # what it can see: the query sources
        ax.text(x0 + 2.2, 30.8, "scoring queries", ha="left", va="center",
                fontsize=11, color=GREY_MID, zorder=3)
        cell, gap, y = 1.5, 0.28, 27.6
        for i in range(8):
            ax.add_patch(Rectangle((x0 + 2.2 + i * (cell + gap), y), cell, 2.2,
                                   facecolor=PALETTE["highlight"], edgecolor=GREY_DARK,
                                   linewidth=0.9, zorder=3))
        ax.text(x0 + 2.2 + 8 * (cell + gap) + 0.9, y + 1.1,
                r"prompt window $\mathcal{Q}_\mathcal{W}$", ha="left", va="center",
                fontsize=11, color=GREY_DARK, zorder=3)
        if is_ours:
            for m, pc in enumerate([PALETTE["blue_secondary"], PALETTE["teal"],
                                    PALETTE["green_3"], PALETTE["violet"]]):
                yy = 24.6 - m * 1.7
                for i in range(6):
                    ax.add_patch(Rectangle((x0 + 2.2 + i * (cell + gap), yy), cell, 1.3,
                                           facecolor=pc, edgecolor=GREY_DARK,
                                           linewidth=0.7, zorder=3))
            ax.text(x0 + 2.2 + 6 * (cell + gap) + 0.9, 22.0,
                    r"$M$ sampled continuations $\mathcal{Q}^{(m)}_{\mathrm{look}}$",
                    ha="left", va="center", fontsize=11, color=GREY_DARK, zorder=3)
        else:
            ax.text(x0 + 2.2, 23.0, "no response-side queries", ha="left", va="center",
                    fontsize=11, color=PALETTE["red_strong"], zorder=3)

        ax.text((x0 + x1) / 2, 15.8, formula, ha="center", va="center",
                fontsize=12.5, color=INK, zorder=3)
        ax.text(x0 + 2.2, 11.7, note, ha="left", va="center", fontsize=11,
                color=GREY_DARK, zorder=3, linespacing=1.5)

        # the measured numbers for THIS criterion, inside its own panel
        if rows:
            B0 = sorted({r["budget"] for r in rows})[0]
            rec, _ = stat(rows, None, B0, meth, recall_metric(rows))
            enr, _ = stat(rows, None, B0, meth, "evidence_enrich")
            bits = []
            if np.isfinite(rec):
                bits.append(f"oracle recall {rec:.3f}")
            if np.isfinite(enr):
                bits.append(f"answer enrichment {enr:.2f}$\\times$")
            if bits:
                ax.add_patch(FancyBboxPatch((x0 + 1.6, 6.3), x1 - x0 - 3.2, 2.3,
                                            boxstyle="round,pad=0,rounding_size=0.7",
                                            linewidth=1.2, edgecolor=col,
                                            facecolor="white", zorder=2))
                ax.text((x0 + x1) / 2, 7.45, "   ".join(bits), ha="center", va="center",
                        fontsize=11.5, color=col, zorder=3)

    for (a, _), (b, _) in zip([(s[1], 0) for s in spans[:-1]], [(s[0], 0) for s in spans[1:]]):
        ax.add_patch(FancyArrowPatch((a + 0.4, 23.0), (b - 0.4, 23.0), arrowstyle="-|>",
                                     mutation_scale=20, linewidth=2.2,
                                     color=PALETTE["neutral"], zorder=4))

    # footer: what is held fixed, and at which budget the numbers were measured
    budgets = sorted({r["budget"] for r in rows}) if rows else []
    ax.add_patch(FancyBboxPatch((1.0, 0.6), 98.0, 4.0,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                linewidth=1.4, edgecolor=PALETTE["blue_secondary"],
                                facecolor="#EEF3F9", zorder=1))
    tail = f"  ·  numbers measured at $B={budgets[0]}$" if budgets else ""
    ax.text(50.0, 2.6,
            "same prompt, prefill and particles  ·  selection held fixed "
            "(uniform per-head chunk top-$B$)  ·  only the scoring rule differs" + tail,
            ha="center", va="center", fontsize=11, color=PALETTE["blue_main"], zorder=3)

    fig.tight_layout(pad=1.0)
    return finalize_figure(fig, out_base, formats=["png", "pdf"], dpi=300, pad=0.06)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run", nargs="*", default=None,
                    help="demo run folder(s) (default: newest under --demos-dir)")
    ap.add_argument("--demos-dir", default=os.path.join(repo, "demos"))
    ap.add_argument("--out-dir", default=None, help="default: <first run>/figures")
    ap.add_argument("--layout", default="combined", choices=["combined", "rows", "cols"],
                    help="ranking_quality arrangement: 'combined' 1x3 with both benchmarks per "
                         "panel, 'rows' 2x3 one benchmark per row, 'cols' 3x2 one benchmark per "
                         "column (matches matlab/make_ranking_quality.m)")
    ap.add_argument("--bins", type=int, default=400,
                    help="position bins in the retention strip (panel b)")
    ap.add_argument("--profile", default=None,
                    help="substring picking which profiles/*.npz feeds the retention strip")
    args = ap.parse_args()

    runs = args.run or [newest_run(args.demos_dir)]
    rows = load_rows(runs)
    out_dir = args.out_dir or os.path.join(runs[0], "figures")
    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=2.0))

    for path in figure_quality(rows, runs, os.path.join(out_dir, "ranking_quality"),
                               layout=args.layout):
        print(f"Saved {path}")
    for path in figure_evidence(rows, runs, os.path.join(out_dir, "ranking_evidence"),
                                args.profile, args.bins):
        print(f"Saved {path}")
    for path in figure_criteria(rows, os.path.join(out_dir, "ranking_criteria")):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
