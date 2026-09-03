"""Reparametrization validation (report stage) - runs V1-V5, D1-D2, S3-S4 on a capture.

No GPU. Reads validation_data.npz from compute_validation.py and prints the table the
paper's validation subsection reports. Every test states what a failure means.

  V1  grand mean of cbar_m, bootstrap CI          CI must contain 0.
      A positive mean means the particles were sampled from a truncated/tempered law
      but centered against the untruncated one, so E[c_u|h_u] != 0. That is a silent
      bias with no other visible symptom -- fix the capture before reading anything else.
  V2  logged R_m vs the formula                    max abs diff < 1e-5.
      An exact T-fold gap means the code used the sequence TOTAL where Eq. 3 uses the
      MEAN, i.e. a T-times sharper softmax than the paper describes.
  V3  lag-k autocorrelation of {c_u}               ~0 for every k.
      Non-zero means truncation leaked in, or the particles entered a repetition
      regime; either way the martingale argument behind V4 does not apply.
  V4  OLS of the empirical across-particle Var(cbar_m) on the predicted (1/T^2) sum_u V_u
      slope ~1. Otherwise Prop. 3 is miscalibrated and only unnormalized cbar is safe.
  V5  across-prompt ratio q95/q05 of s_X = sqrt(Vbar/T)
      < 1.3 -> keep NORMALIZE_TYPICALITY=0 and demote Prop. 3 to a remark.
      > 2   -> the normalization is load-bearing; promote it.
  D1  decomposition of Var_m(L), L = cbar - Hbar (the log-probability term the old
      softmax actually saw): Var(cbar) + Var(Hbar) - 2Cov(cbar, Hbar).
      If Var(Hbar) dominates, the draft's "log-probability weighting" was operationally
      an entropy penalty and separating the channels is necessary, not cosmetic.
  D2  within-prompt Spearman(cbar_m, Hbar_m).
  S3  Spearman of each candidate signal with per-particle oracle utility, sign-oriented.
      The key comparison is cbar_m vs raw log p: if cbar wins, removing the confound
      demonstrably improves the per-particle signal.
  S4  incremental value of disagreement: semipartial Spearman of D_m with the residual
      of utility on (cbar_m, Hbar_m), plus delta-R^2 and a within-prompt permutation
      null. This is the GATE for carrying lambda_D > 0 into the end-to-end grid.
"""
import argparse
import os

import numpy as np


# ---------------------------------------------------------------- small stats
def _rank(a):
    order = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    return order


def spearman(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else 0.0


def pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def bootstrap_ci(x, n=10000, alpha=0.05, rng=None):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def ols(x, y):
    """-> (slope, intercept, R^2) for a single regressor."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(coef[0]), float(coef[1]), float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def residualize(y, X):
    """Residual of y after least-squares projection onto [X, 1]."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != y.shape[0]:
        X = X.T
    A = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef, float(1 - ((y - A @ coef) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


# ---------------------------------------------------------------- the tests
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="validation_data*.npz")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    c, H, V = d["c"], d["H"], d["V"]                 # [P, M, T]
    P, M, T = c.shape
    beta, gamma, lambda_d = float(d["beta"]), float(d["gamma"]), float(d["lambda_d"])
    Dz = d["Dz"]                                      # [P, M]
    util = d["util"]                                  # [P, M] (may be all-nan)
    c_bar, H_bar, V_bar = c.mean(-1), H.mean(-1), V.mean(-1)   # [P, M]
    rng = np.random.default_rng(0)

    print(f"\n{'='*78}\nReparametrization validation   "
          f"P={P} prompts  M={M} particles  T={T} steps\n"
          f"  (beta, gamma, lambda_D) = ({beta:g}, {gamma:g}, {lambda_d:g})   "
          f"temperature={float(d['temperature']):g}  "
          f"top_k={d['top_k']}  top_p={d['top_p']}\n{'='*78}")

    # -- V1: is the centering law the sampling law? -------------------------
    lo, hi = bootstrap_ci(c_bar, args.bootstrap, rng=rng)
    mean_c = float(np.nanmean(c_bar))
    v1 = "PASS" if lo <= 0.0 <= hi else "FAIL"
    print(f"\nV1  E[cbar_m] = {mean_c:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]   {v1}")
    if v1 == "FAIL":
        print("    -> positive mean = truncation/temperature mismatch. The particles were "
              "sampled\n       from a law c_u is NOT centered against. Fix the capture "
              "(step_stats must see\n       the sampler's temperature/top_k/top_p) and "
              "re-capture before reading anything else.")

    # -- V2: does the logged R match the formula? ---------------------------
    if "R" in d:
        rhs = beta * c_bar - gamma * H_bar - lambda_d * Dz
        gap = float(np.nanmax(np.abs(d["R"] - rhs)))
        print(f"\nV2  max |R_logged - (beta*cbar - gamma*Hbar - lambda_D*Dhat)| = {gap:.3e}   "
              f"{'PASS' if gap < 1e-5 else 'FAIL'}")
        if gap >= 1e-5:
            ratio = float(np.nanmedian(np.abs(d["R"]) / np.maximum(np.abs(rhs), 1e-9)))
            print(f"    -> median |R_logged / formula| = {ratio:.3f}"
                  + (f"; ~T = {T} means the code used the sequence TOTAL, not the mean."
                     if abs(ratio - T) < 0.5 * T else ""))

    # -- V3: are the c_u serially uncorrelated? ------------------------------
    print("\nV3  lag-k autocorrelation of {c_u}   (martingale differences => ~0 at every k)")
    for k in range(1, min(T, 6)):
        r = pearson(c[:, :, :-k].ravel(), c[:, :, k:].ravel())
        print(f"      lag {k}: rho = {r:+.4f}" + ("   <-- non-negligible" if abs(r) > 0.1 else ""))

    # -- V4: does Prop. 3 predict the observed spread? -----------------------
    emp = c_bar.var(axis=1, ddof=1)                       # per prompt, across particles
    pred = V.mean(axis=(1, 2)) / T
    slope, _, r2 = ols(pred, emp)
    print(f"\nV4  OLS( empirical Var_m(cbar) ~ (1/T) * mean V_u ):  "
          f"slope = {slope:+.3f}  R^2 = {r2:.3f}   "
          f"{'PASS' if 0.7 <= slope <= 1.4 else 'CHECK'}")
    if not (0.7 <= slope <= 1.4):
        print("    -> Prop. 3 is miscalibrated on these captures; report the unnormalized "
              "cbar only.")

    # -- V5: is a single global softmax scale calibrated across prompts? -----
    s = np.sqrt(V.mean(-1) / T).mean(-1)                  # [P]
    q05, q95 = np.quantile(s, 0.05), np.quantile(s, 0.95)
    ratio = float(q95 / max(q05, 1e-12))
    verdict = ("keep NORMALIZE_TYPICALITY=0 (demote Prop. 3 to a remark)" if ratio < 1.3
               else "normalization is load-bearing: promote it" if ratio > 2.0
               else "inconclusive band; report the ratio and keep the switch off by default")
    print(f"\nV5  s_X = sqrt(Vbar/T):  q05 = {q05:.4f}  q95 = {q95:.4f}  "
          f"ratio = {ratio:.2f}\n    -> {verdict}")

    # -- D1/D2: how much of the old score's spread was entropy? --------------
    L = c_bar - H_bar                                     # the raw log-probability term
    var_L = L.var(1, ddof=1)
    var_c = c_bar.var(1, ddof=1)
    var_H = H_bar.var(1, ddof=1)
    cov = ((c_bar - c_bar.mean(1, keepdims=True)) *
           (H_bar - H_bar.mean(1, keepdims=True))).sum(1) / max(M - 1, 1)
    ok = var_L > 1e-12
    shares = np.stack([var_c[ok], var_H[ok], -2 * cov[ok]], -1) / var_L[ok, None]
    sh = shares.mean(0)
    print(f"\nD1  Var_m(L) = Var(cbar) + Var(Hbar) - 2Cov,  L = cbar - Hbar "
          f"(the term the OLD softmax saw)")
    print(f"      Var(cbar) share = {sh[0]:+.3f}   Var(Hbar) share = {sh[1]:+.3f}   "
          f"-2Cov share = {sh[2]:+.3f}")
    print("    -> " + ("Var(Hbar) dominates: the draft's 'log-probability weighting' was "
                       "operationally an\n       entropy penalty, so separating the channels "
                       "is necessary, not cosmetic."
                       if sh[1] > sh[0] else
                       "Var(cbar) dominates: the confound is mild. Report that honestly and "
                       "keep the\n       change for interpretability."))
    rho_wp = np.array([spearman(c_bar[p], H_bar[p]) for p in range(P)])
    print(f"D2  within-prompt Spearman(cbar_m, Hbar_m): mean {np.nanmean(rho_wp):+.3f} "
          f"(median {np.nanmedian(rho_wp):+.3f})")

    # -- S3/S4: does any of this predict oracle cache utility? ---------------
    if np.all(~np.isfinite(util)):
        print("\nS3/S4  skipped: no per-particle utility in this capture "
              "(re-run compute_validation.py with --utility).")
        print(f"{'='*78}\n")
        return

    logp_bar = c_bar - H_bar
    signals = {
        "log p_theta (raw)": logp_bar,
        "cbar_m (centered)": c_bar,
        "-Hbar_m": -H_bar,
        "-Dhat_m": -Dz,
        f"R_m /E (1,{gamma:g},0)": c_bar - gamma * H_bar,
        f"R_m no-entropy (1,0,{lambda_d or 0.3:g})": c_bar - (lambda_d or 0.3) * Dz,
    }
    print("\nS3  Spearman(signal, per-particle oracle utility), sign-oriented "
          "(higher = better particle)")
    for name, sig in signals.items():
        print(f"      {name:<34s} rho = {spearman(sig, util):+.4f}")
    print("    -> the comparison that matters is cbar_m vs raw log p_theta: if cbar wins, "
          "removing\n       the entropy confound improves the per-particle signal itself.")

    # S4: semipartial correlation + delta-R^2 + within-prompt permutation null.
    u = util.ravel()
    base = np.stack([c_bar.ravel(), H_bar.ravel()], 1)
    resid_u, r2_base = residualize(u, base)
    resid_D, _ = residualize(Dz.ravel(), base)
    semipartial = spearman(resid_D, resid_u)
    _, r2_full = residualize(u, np.concatenate([base, Dz.ravel()[:, None]], 1))
    d_r2 = r2_full - r2_base

    null = np.empty(args.permutations)
    for i in range(args.permutations):
        perm = np.concatenate([rng.permutation(Dz[p]) for p in range(P)])  # within prompt
        r_perm, _ = residualize(perm, base)
        null[i] = spearman(r_perm, resid_u)
    pval = float((np.abs(null) >= abs(semipartial)).mean())

    print(f"\nS4  incremental value of disagreement, controlling for (cbar_m, Hbar_m)")
    print(f"      semipartial Spearman(D_m, utility | cbar, Hbar) = {semipartial:+.4f}")
    print(f"      R^2: (cbar,Hbar) = {r2_base:.4f} -> +Dhat = {r2_full:.4f}   "
          f"delta = {d_r2:+.4f}")
    print(f"      within-prompt permutation null ({args.permutations} draws): p = {pval:.4f}")
    print("    -> " + ("non-zero and sign-consistent across benchmarks: carry lambda_D > 0 "
                       "into the\n       end-to-end grid (Stage 2)."
                       if pval < 0.05 else
                       "near-zero: a reportable NEGATIVE CONTROL -- disagreement is subsumed "
                       "by the\n       model's own conditional uncertainty structure. Keep "
                       "lambda_D = 0 and document\n       the switch. Run this per benchmark; "
                       "a pooled number would hide a sign flip."))
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
