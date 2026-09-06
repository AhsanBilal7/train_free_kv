#!/usr/bin/env python3
"""Canonical naming for a run's output folder under experiments/<BENCHMARK>/pred/.

ONE source of truth, used three ways:
  * generation -- experiments/*/run.sh calls this as a CLI to build `out_name`
  * parsing    -- collect_results.py / eval.py import parse_run_name()
  * migration  -- scripts/rename_pred_runs.py rewrites pre-existing folders

Scheme for `causal_lookahead` runs (mirrors the config file names exactly):

    lorekv-<cell>-<brule>-<sel>[-<model>]_B_<B>_chunk_<cs>_M_<M>_T_<T>
                                          _lL_<lL>_lT_<lT>_nrm_<0|1>
                                          _alpha_<alpha>_kappa_<kappa>
                                          [_obj_<unprojected|attention>]

The leading, hyphenated part is the run's IDENTITY and is the same string as its
config file (config/lorekv-<cell>-<brule>-<sel>[-llama].conf). The trailing,
underscored part is everything the config name does not pin down, so two runs that
differ in any of it never collide.

    <cell>   the Eq. 3 reliability coefficients (beta, gamma, lambda_D)
    <brule>  head-budget rule from tau:      unif (0) | adap (1) | adap<tau>
    <sel>    selection rule from lambda_k/v: topk (0) | div (0.1) | div<lk>-<lv>
    <model>  '' for Mistral-7B-Instruct-v0.2, '-llama' for Llama-3.1-8B-Instruct,
             '-qwen2' for Qwen2.5-14B-Instruct, '-ministral3' for
             Ministral-3-8B-Instruct-2512

IMPORTANT: <cell>, <brule> and <sel> are derived from the run's ACTUAL numeric
hyper-parameters, never from the config file name. A sweep such as
`LAMBDA_D="0.0 0.3" CONFIG_FILE=config/lorekv-typ-dis-unif-topk.conf` therefore
writes `lorekv-typ-...` and `lorekv-typ-dis-...` folders, and the folder name can
never disagree with what was actually run. Coefficients that match no named cell
fall back to `b<beta>g<gamma>lD<lambdaD>`, which is still unique and parseable.

Other modes (andpro / critical / causal_forward_only) keep their existing
`<mode>_<model-tag>_<B>_chunk_<cs>_red_<red>` scheme; both are parsed.
"""
import argparse
import re

# --- (beta, gamma, lambda_D) -> cell name; see README "The reliability cells" ---
CELLS = {
    (0.0, 0.0, 0.0): "uniform",    # the missing control: omega_m = 1/M
    (1.0, 0.0, 0.0): "typ",        # pure typicality; the real no-entropy control
    (1.0, 1.0, 0.0): "logp",       # the old "log-probability only" (gamma = 1!)
    (1.0, 1.3, 0.0): "ent",        # the old LORE-KV /E core
    (0.0, 1.0, 0.0): "ent-only",   # pure entropy
    (1.0, 1.0, 0.3): "dis",        # the old LORE-KV /D core (entropy AND disagreement)
    (1.0, 0.0, 0.3): "typ-dis",    # disagreement INSTEAD of entropy
}
CELL_COEFFS = {v: k for k, v in CELLS.items()}

# Model tag -> short suffix. A tag that extends a known base keeps its remainder,
# so the three-seed runs (MODEL_TAG=Mistral-7B-Instruct-v0.2-s0) become '-s0'.
MODEL_BASES = [("Mistral-7B-Instruct-v0.2", ""), ("Llama-3.1-8B-Instruct", "-llama"),
               # The extra backbones; the short forms match config/*-qwen2.conf and
               # config/*-ministral3.conf, keeping the invariant that a run folder's
               # hyphenated head IS its config-file basename.
               ("Qwen2.5-14B-Instruct", "-qwen2"), ("Ministral-3-8B-Instruct-2512", "-ministral3")]

_TOL = 1e-9


def _num(x):
    return float(x)


def _fmt(x):
    """Compact, round-trippable number: 1.0 -> '1', 0.30 -> '0.3'."""
    f = float(x)
    if f == int(f):
        return str(int(f))
    return repr(round(f, 6)).rstrip("0").rstrip(".")


def cell_name(beta, gamma, lambda_d):
    key = (_num(beta), _num(gamma), _num(lambda_d))
    for k, v in CELLS.items():
        if all(abs(a - b) < 1e-6 for a, b in zip(key, k)):
            return v
    return f"b{_fmt(key[0])}g{_fmt(key[1])}lD{_fmt(key[2])}"


def budget_rule_name(tau):
    t = _num(tau)
    if abs(t) < _TOL:
        return "unif"
    if abs(t - 1.0) < _TOL:
        return "adap"
    return f"adap{_fmt(t)}"


def selection_name(lambda_k, lambda_v):
    lk, lv = _num(lambda_k), _num(lambda_v)
    if abs(lk) < _TOL and abs(lv) < _TOL:
        return "topk"
    if abs(lk - 0.1) < _TOL and abs(lv - 0.1) < _TOL:
        return "div"
    return f"div{_fmt(lk)}-{_fmt(lv)}"


def model_suffix(model_tag):
    for base, short in MODEL_BASES:
        if model_tag == base:
            return short
        if model_tag.startswith(base + "-"):
            return short + "-" + model_tag[len(base) + 1:]
    return "-" + model_tag


def model_tag_from_suffix(suffix):
    """Inverse of model_suffix. '' -> Mistral, '-llama' -> Llama, else verbatim."""
    for base, short in MODEL_BASES:
        if suffix == short:
            return base
        if short and suffix.startswith(short + "-"):
            return base + "-" + suffix[len(short) + 1:]
    if suffix == "":
        return MODEL_BASES[0][0]
    return suffix.lstrip("-")


def config_name(beta, gamma, lambda_d, tau, lambda_k, lambda_v, model_tag):
    """The run's identity: identical to its config file's basename."""
    return (f"lorekv-{cell_name(beta, gamma, lambda_d)}"
            f"-{budget_rule_name(tau)}-{selection_name(lambda_k, lambda_v)}"
            f"{model_suffix(model_tag)}")


def build_run_name(*, mode, model_tag, budget, chunk_size,
                   num_particles=None, lookahead_len=None, lambda_lookahead=None,
                   lookahead_temperature=None, beta=None, gamma=None, lambda_d=None,
                   normalize_typicality=None, robust_alpha=None, robust_topk_beta=None,
                   lambda_k=None, lambda_v=None, budget_tau=None, lambda_red=None,
                   score_objective=None):
    if mode != "causal_lookahead":
        return f"{mode}_{model_tag}_{budget}_chunk_{chunk_size}_red_{lambda_red}"
    ident = config_name(beta, gamma, lambda_d, budget_tau, lambda_k, lambda_v, model_tag)
    name = (f"{ident}"
            f"_B_{budget}_chunk_{chunk_size}"
            f"_M_{num_particles}_T_{lookahead_len}"
            f"_lL_{lambda_lookahead}_lT_{lookahead_temperature}"
            f"_nrm_{normalize_typicality}"
            f"_alpha_{robust_alpha}_kappa_{robust_topk_beta}")
    # App. C (iii) perturbation objective. Appended ONLY for the two ablation values, so
    # every folder written before this switch existed keeps its exact name and no
    # already-collected result has to be re-run or renamed.
    if score_objective not in (None, "", "projected"):
        name += f"_obj_{score_objective}"
    return name


# --------------------------------------------------------------------------
# Parsing. Handles the current scheme and both pre-reparametrization schemes,
# so collect_results.py keeps working on folders written before this change.
# --------------------------------------------------------------------------
_IDENT_RE = re.compile(
    r"^lorekv-(?P<cell>.+?)"
    r"-(?P<brule>unif|adap(?:[0-9.]+)?)"
    r"-(?P<sel>topk|div(?:[0-9.]+-[0-9.]+)?)"
    r"(?P<model>-.*)?$")

# legacy: <mode>_<model>_<B>_chunk_<cs><tail>
_LEGACY_RE = re.compile(
    r"^(?P<mode>.+)_(?P<model>[^_]+)_(?P<budget>\d+)_chunk_(?P<chunk>[^_]+)(?P<tail>.*)$")


def _pairs(tail):
    out = {}
    toks = tail.strip("_").split("_")
    for i in range(0, len(toks) - 1, 2):
        out[toks[i]] = toks[i + 1]
    return out


def parse_run_name(name):
    """Folder name -> flat dict of hyper-parameters, or None if unparseable.

    Always populated when it succeeds: mode, model, budget, chunk. For
    causal_lookahead runs also: cell, brule, sel, b, g, lD, tau, lK, lV, M, T,
    lL, lT, nrm, alpha, kappa. Legacy folders are translated into the SAME keys
    via Proposition 1, (beta, gamma, lambda_D) = (eta, eta*(1+lambda_H), eta*lambda_D),
    so a table can mix old and new folders without special-casing.
    """
    name = name[len("multi_"):] if name.startswith("multi_") else name

    if name.startswith("lorekv-"):
        ident, _, tail = name.partition("_")
        m = _IDENT_RE.match(ident)
        if not m:
            return None
        p = _pairs(tail)
        cell, brule, sel = m.group("cell"), m.group("brule"), m.group("sel")
        coeffs = CELL_COEFFS.get(cell)
        if coeffs is None:
            g = re.match(r"^b([0-9.]+)g([0-9.]+)lD([0-9.]+)$", cell)
            coeffs = tuple(float(x) for x in g.groups()) if g else (None, None, None)
        if brule == "unif":
            tau = "0.0"
        elif brule == "adap":
            tau = "1.0"
        else:
            tau = brule[len("adap"):]
        if sel == "topk":
            lk = lv = "0.0"
        elif sel == "div":
            lk = lv = "0.1"
        else:
            lk, lv = sel[len("div"):].split("-")
        out = {"mode": "causal_lookahead",
               "model": model_tag_from_suffix(m.group("model") or ""),
               "cell": cell, "brule": brule, "sel": sel,
               "b": coeffs[0], "g": coeffs[1], "lD": coeffs[2],
               "tau": tau, "lK": lk, "lV": lv,
               "budget": p.get("B"), "chunk": p.get("chunk")}
        for k in ("M", "T", "lL", "lT", "nrm", "alpha", "kappa"):
            out[k] = p.get(k)
        # Absent for every core run; 'projected' is the implicit default (see build_run_name).
        out["obj"] = p.get("obj", "projected")
        return out

    m = _LEGACY_RE.match(name)
    if not m:
        return None
    out = {"mode": m.group("mode"), "model": m.group("model"),
           "budget": m.group("budget"), "chunk": m.group("chunk")}
    out.update(_pairs(m.group("tail")))
    if out["mode"] == "causal_lookahead":
        # Proposition 1: the (lambda_H, eta) parametrization is the ray
        # (beta, gamma, lambda_D) = (eta, eta*(1 + lambda_H), eta*lambda_D).
        if "b" not in out and "lH" in out:
            eta = float(out.get("eta", 1.0))
            lh = float(out.get("lH", 0.0))
            ld = float(out.get("lD", 0.0))
            out["b"], out["g"], out["lD"] = eta, eta * (1.0 + lh), eta * ld
            out["nrm"] = out.get("nrm", "0")
        if "kappa" not in out and "beta" in out:      # old name for the top-fraction
            out["kappa"] = out.pop("beta")
        if out.get("b") is not None:
            out["cell"] = cell_name(out["b"], out["g"], out["lD"])
            out["brule"] = budget_rule_name(out.get("tau", 0.0))
            out["sel"] = selection_name(out.get("lK", 0.0), out.get("lV", 0.0))
    return out


def canonical_name(old_name):
    """Old folder name -> its name under the current scheme (None if not applicable)."""
    p = parse_run_name(old_name)
    if not p or p.get("mode") != "causal_lookahead" or p.get("b") is None:
        return None
    prefix = "multi_" if old_name.startswith("multi_") else ""
    return prefix + build_run_name(
        mode="causal_lookahead", model_tag=p["model"], budget=p["budget"],
        chunk_size=p["chunk"], num_particles=p.get("M"), lookahead_len=p.get("T"),
        lambda_lookahead=p.get("lL"), lookahead_temperature=p.get("lT"),
        beta=p["b"], gamma=p["g"], lambda_d=p["lD"],
        normalize_typicality=p.get("nrm", "0"),
        robust_alpha=p.get("alpha"), robust_topk_beta=p.get("kappa"),
        lambda_k=p["lK"], lambda_v=p["lV"], budget_tau=p["tau"],
        score_objective=p.get("obj"))


def main():
    ap = argparse.ArgumentParser(description="Print a run's canonical output-folder name.")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--budget", required=True)
    ap.add_argument("--chunk-size", required=True)
    ap.add_argument("--num-particles"); ap.add_argument("--lookahead-len")
    ap.add_argument("--lambda-lookahead"); ap.add_argument("--lookahead-temperature")
    ap.add_argument("--beta"); ap.add_argument("--gamma"); ap.add_argument("--lambda-d")
    ap.add_argument("--normalize-typicality")
    ap.add_argument("--robust-alpha"); ap.add_argument("--robust-topk-beta")
    ap.add_argument("--lambda-k"); ap.add_argument("--lambda-v")
    ap.add_argument("--budget-tau"); ap.add_argument("--lambda-red")
    ap.add_argument("--score-objective")
    a = ap.parse_args()
    print(build_run_name(
        mode=a.mode, model_tag=a.model_tag, budget=a.budget, chunk_size=a.chunk_size,
        num_particles=a.num_particles, lookahead_len=a.lookahead_len,
        lambda_lookahead=a.lambda_lookahead, lookahead_temperature=a.lookahead_temperature,
        beta=a.beta, gamma=a.gamma, lambda_d=a.lambda_d,
        normalize_typicality=a.normalize_typicality,
        robust_alpha=a.robust_alpha, robust_topk_beta=a.robust_topk_beta,
        lambda_k=a.lambda_k, lambda_v=a.lambda_v, budget_tau=a.budget_tau,
        lambda_red=a.lambda_red, score_objective=a.score_objective))


if __name__ == "__main__":
    main()
