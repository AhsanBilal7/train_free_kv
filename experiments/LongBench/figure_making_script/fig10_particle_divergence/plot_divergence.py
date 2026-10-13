"""Figure 10 (plot stage) - reads divergence_data.npz.

A  D_m vs utility     : one point per lookahead particle. A DOWNWARD trend means a
                        particle whose future-query trajectory disagrees with the
                        ensemble is a worse proxy for the true decode queries --
                        the premise for penalizing D_m in R_m (Eq. 7).
B  entropy vs utility : the same particles against the signal D_m replaces. Plotted
                        on its own axes so the two fits can be read side by side.
C  correlation bars   : Spearman(signal, utility) for log-prob, -entropy, -D_m and
                        the two combined confidences. This is the panel that says
                        which term actually carries information about usefulness.
D  weighting outcome  : ensemble coverage of Q_future under uniform / log-prob /
                        log-prob-entropy / log-prob-divergence weights, plus the
                        per-sample win rate of the divergence rule over the entropy
                        rule (same particles, so weights are the only difference).
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

# Rules are labelled by their Eq. 3 coordinates (beta, gamma, lambda_D); (1,0,0) is
# the no-entropy control and (1,0,lambda_D) uses disagreement without entropy.
LABEL = {"uniform": "Uniform $(0,0,0)$", "logprob": "log-prob $(1,1,0)$",
         "typicality": "typicality $(1,0,0)$",
         "entropy": "/E $(1,\\gamma,0)$", "divergence": "/D $(1,1,\\lambda_D)$",
         "typ_div": "no-entropy $(1,0,\\lambda_D)$"}
COLOR = {"uniform": "#2ca02c", "logprob": "#ff7f0e", "typicality": "#9467bd",
         "entropy": "#1f77b4", "divergence": "#8c564b", "typ_div": "#d62728"}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else 0.0


def _scatter_fit(ax, x, y, color, xlabel, ylabel, title):
    ax.scatter(x, y, s=16, alpha=0.45, color=color, edgecolor="none")
    if len(x) > 2 and np.ptp(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(xs, m * xs + b, color="#333333", lw=2,
                label=f"slope {m:+.3f}\nPearson r={r:+.2f}\nSpearman $\\rho$={_spearman(x, y):+.2f}")
        ax.legend(fontsize=7, loc="best")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="explicit .npz (overrides auto-pick of newest tagged)")
    ap.add_argument("--data-dir", default=None, help="folder to search for the newest tagged .npz")
    ap.add_argument("--utility", default="recall", choices=["recall", "align"],
                    help="oracle utility plotted on the y-axis / used for the correlations")
    ap.add_argument("--out", default=None, help="explicit .png (default: next to the .npz, same tag)")
    args = ap.parse_args()
    args.data = common.pick_npz(args.data, args.data_dir, HERE, "divergence_data")
    args.out = common.png_path_for(args.data, "fig10_particle_divergence", args.out)
    d = np.load(args.data, allow_pickle=True)

    util = d["util_recall"] if args.utility == "recall" else d["util_align"]
    ulabel = "Oracle utility (recall@B)" if args.utility == "recall" else "Oracle utility (coverage)"
    gamma_E = float(d["gamma_E"]) if "gamma_E" in d else 1.0 + float(d["lambda_h"])
    lambda_d = float(d["lambda_d"])
    # cbar_m = logp_mean + H_mean, recoverable from any capture (old ones included).
    c_bar = d["c_bar"] if "c_bar" in d else (d["logprob"] + d["entropy"])

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    ax1, ax2, ax3, ax4 = axes

    # A / B: the relationship itself, disagreement and entropy on identical particles.
    _scatter_fit(ax1, d["divergence"], util, "#d62728",
                 "Trajectory disagreement $D_m$", ulabel,
                 "Atypical trajectories are worse proxies")
    _scatter_fit(ax2, d["entropy"], util, "#1f77b4",
                 "Predictive entropy $\\bar{H}_m$", ulabel,
                 "The signal $D_m$ replaces")

    # C: which signal predicts usefulness (sign-oriented: higher = better particle).
    # S3: does removing the entropy confound from the log-probability channel improve
    # the per-particle signal? The comparison that matters is cbar_m vs raw log p.
    signals = {
        "$\\log p_\\theta$": d["logprob"],
        "$\\bar{c}_m$ (centered)": c_bar,
        "$-\\bar{H}_m$": -d["entropy"],
        "$-\\hat{D}_m$": -d["divergence_z"],
        f"$R_m$ /E\n$(1,{gamma_E:g},0)$": c_bar - gamma_E * d["entropy"],
        f"$R_m$ no-entropy\n$(1,0,{lambda_d:g})$": c_bar - lambda_d * d["divergence_z"],
    }
    names = list(signals)
    corr = [_spearman(signals[k], util) for k in names]
    bar_colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    ax3.bar(np.arange(len(names)), corr, color=bar_colors)
    ax3.set_xticks(np.arange(len(names)))
    ax3.set_xticklabels(names, fontsize=7, rotation=20, ha="right")
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_ylabel(f"Spearman(signal, {args.utility})")
    ax3.set_title("Which signal predicts utility", fontsize=10)
    ax3.grid(True, axis="y", alpha=0.3)

    # D: does the weighting built on it actually cover Q_future better?
    # The bars compare MEANS, and the across-prompt spread is an order of magnitude
    # larger than the between-rule difference, so the error bars are the standard error
    # of the mean (not the sample std, which would say nothing about the comparison).
    # The comparison is paired -- every rule saw the identical particles on every
    # sample -- so the per-sample win rate is reported alongside.
    keys = [str(k) for k in d["strat_keys"]]
    means = d["strat_mean"]; stds = d["strat_std"]
    n_s = int(d["n_samples"]) if "n_samples" in d else 1
    sems = stds / max(1.0, np.sqrt(n_s))
    x = np.arange(len(keys))
    ax4.bar(x, means, yerr=sems, capsize=3, color=[COLOR.get(k, "#333") for k in keys], alpha=0.9)
    ax4.set_xticks(x)
    ax4.set_xticklabels([LABEL.get(k, k) for k in keys], fontsize=7, rotation=20, ha="right")
    ax4.set_ylabel("Coverage of $Q_{future}$  (bars: $\\pm$1 s.e.m.)")
    ax4.set_title("Ensemble coverage by weighting rule", fontsize=10)
    ax4.grid(True, axis="y", alpha=0.3)
    lo = float(np.min(means - sems)); hi = float(np.max(means + sems))
    if hi > lo:
        pad = 0.45 * (hi - lo)
        ax4.set_ylim(max(0.0, lo - pad), hi + pad)
    if "strat_per_sample" in d and "entropy" in keys and "divergence" in keys:
        per = d["strat_per_sample"]
        ent_s = per[keys.index("entropy")]; div_s = per[keys.index("divergence")]
        if len(ent_s):
            win = float(np.mean(div_s > ent_s))
            delta = float(np.mean(div_s - ent_s))
            ax4.text(0.5, 0.97,
                     f"paired vs entropy rule: $\\Delta$={delta:+.4f}, "
                     f"wins on {win*100:.0f}% of {len(ent_s)} samples",
                     transform=ax4.transAxes, ha="center", va="top", fontsize=7)

    n_p = len(util)
    fig.suptitle("Trajectory disagreement as the particle confidence signal "
                 f"(B={int(d['budget'])}, {d['dataset']}, {n_p} particles)", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
