#!/usr/bin/env python3
"""Names of the run folders in ``experiments/<benchmark>/pred/``.

``scripts/sweep.py`` makes each folder name with :func:`build_run_name`. The
evaluation and table scripts get the hyperparameters back with
:func:`parse_run_name`.

LORE-KV runs (``--mode causal_lookahead``)::

    lorekv-<cell>-<brule>-<sel>[-<model>]_B_<B>_chunk_<cs>_M_<M>_T_<T>
        _lL_<lambda_L>_lT_<temperature>_nrm_<0|1>_alpha_<alpha>_kappa_<kappa>
        [_obj_<unprojected|attention|andpro>]

The part before ``_B_`` identifies the run. It is equal to the name of the config
file (``config/lorekv-ent-unif-topk-llama.conf`` writes
``lorekv-ent-unif-topk-llama_B_...``):

    <cell>   Reliability coefficients (beta, gamma, lambda_D) of Eq. 3; see CELLS.
    <brule>  Head-budget rule from tau (Eq. 11): unif (0), adap (1) or adap<tau>.
    <sel>    Selection rule from (lambda_k, lambda_v) (App. D-C): topk (0, 0),
             div (0.1, 0.1) or div<lk>-<lv>.
    <model>  '' for Mistral-7B-Instruct-v0.2, '-llama', '-qwen2', '-ministral3',
             else the model tag.

The name comes from the values that the run uses, not from the config file name.
A command-line sweep (``LAMBDA_D="0.0 0.3"``) thus writes one folder for each
cell. Coefficients that are not in CELLS give ``b<beta>g<gamma>lD<lambda_D>``.

``RUN_NAME_PREFIX`` replaces the identifier. For example
``config/andpro-compute-eq.conf`` writes ``andpro-compute-eq_B_...``.
CUSTOM_PREFIXES gives the coefficients of each prefix.

Other methods write ``<mode>_<model-tag>_<B>_chunk_<cs>``. The parser also reads
older folders that end in ``_red_<x>``.

CriticalKV (``--mode critical``) writes ``criticalkv_<model-tag>_<B>_chunk_<cs>``.
Older ``critical_*`` folders come from a ``critical`` mode that ran the Ada-KV
cluster without the value term. The parser thus returns ``mode='ada'`` and
``legacy_critical=True`` for them, and they are not mixed with CriticalKV runs.
"""
import argparse
import re

# (beta, gamma, lambda_D) of Eq. 3 -> cell name. The four settings of Table 4 are
# ent (LORE-KV /E), dis (LORE-KV /D), logp (log-probability only: since
# mean_u log q = c_bar - H_bar it is (1, 1, 0)) and ent-dis; the others are
# additional controls (uniform weights, typicality only, entropy only).
CELLS = {
    (0.0, 0.0, 0.0): "uniform",    # omega_m = 1/M
    (1.0, 0.0, 0.0): "typ",        # centered typicality only
    (1.0, 1.0, 0.0): "logp",       # log-probability only
    (1.0, 1.3, 0.0): "ent",        # LORE-KV /E
    (0.0, 1.0, 0.0): "ent-only",   # entropy only
    (1.0, 1.0, 0.3): "dis",        # LORE-KV /D
    (1.0, 0.0, 0.3): "typ-dis",    # disagreement without entropy
    (1.0, 1.3, 0.3): "ent-dis",    # entropy + disagreement control (Table 4)
}
CELL_COEFFS = {v: k for k, v in CELLS.items()}

# name_prefix -> (beta, gamma, lambda_d, tau, lambda_k, lambda_v) the prefix stands
# in for. Registered here so parse_run_name can recover the run's actual numeric
# identity from a folder written with an overridden prefix (build_run_name's
# name_prefix kwarg / RUN_NAME_PREFIX), the same way it does for the auto-derived
# `lorekv-...` identity.
CUSTOM_PREFIXES = {
    # config/andpro-compute-eq.conf: uniform cell (omega_m = 1/M), unif budgets, top-K.
    "andpro-compute-eq": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

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
    """Return a short number string that converts back to the same value: 1.0 -> '1', 0.30 -> '0.3'."""
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
    """Return the model tag of a name suffix (the inverse of :func:`model_suffix`)."""
    for base, short in MODEL_BASES:
        if suffix == short:
            return base
        if short and suffix.startswith(short + "-"):
            return base + "-" + suffix[len(short) + 1:]
    if suffix == "":
        return MODEL_BASES[0][0]
    return suffix.lstrip("-")


def config_name(beta, gamma, lambda_d, tau, lambda_k, lambda_v, model_tag):
    """Return the identifier of a LORE-KV run, which is equal to its config file name."""
    return (f"lorekv-{cell_name(beta, gamma, lambda_d)}"
            f"-{budget_rule_name(tau)}-{selection_name(lambda_k, lambda_v)}"
            f"{model_suffix(model_tag)}")


def build_run_name(*, mode, model_tag, budget, chunk_size,
                   num_particles=None, lookahead_len=None, lambda_lookahead=None,
                   lookahead_temperature=None, beta=None, gamma=None, lambda_d=None,
                   normalize_typicality=None, robust_alpha=None, robust_topk_beta=None,
                   lambda_k=None, lambda_v=None, budget_tau=None,
                   score_objective=None, name_prefix=None):
    """Return the folder name of one run. See the module docstring for the format."""
    if mode != "causal_lookahead":
        label = "criticalkv" if mode == "critical" else mode
        return f"{label}_{model_tag}_{budget}_chunk_{chunk_size}"
    # RUN_NAME_PREFIX / --name-prefix: override just the identity string. Everything
    # else -- the numeric tail, obj suffix -- is unchanged, so the folder still
    # encodes exactly what was run; only the leading label differs from what the
    # numbers would otherwise derive (see CUSTOM_PREFIXES / module docstring).
    ident = (name_prefix + model_suffix(model_tag)) if name_prefix else \
        config_name(beta, gamma, lambda_d, budget_tau, lambda_k, lambda_v, model_tag)
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
# Parsing
# --------------------------------------------------------------------------
_IDENT_RE = re.compile(
    r"^lorekv-(?P<cell>.+?)"
    r"-(?P<brule>unif|adap(?:[0-9.]+)?)"
    r"-(?P<sel>topk|div(?:[0-9.]+-[0-9.]+)?)"
    r"(?P<model>-.*)?$")

# every other method: <mode>_<model>_<B>_chunk_<cs>[_red_<x>]
_BASELINE_RE = re.compile(
    r"^(?P<mode>.+)_(?P<model>[^_]+)_(?P<budget>\d+)_chunk_(?P<chunk>[^_]+)(?P<tail>.*)$")


def _pairs(tail):
    out = {}
    toks = tail.strip("_").split("_")
    for i in range(0, len(toks) - 1, 2):
        out[toks[i]] = toks[i + 1]
    return out


def parse_run_name(name):
    """Return the hyperparameters of a run folder as a dict, or None.

    The dict always has ``mode``, ``model``, ``budget`` and ``chunk``. A LORE-KV run
    also has ``cell``, ``brule``, ``sel``, ``b``, ``g``, ``lD``, ``tau``, ``lK``,
    ``lV``, ``M``, ``T``, ``lL``, ``lT``, ``nrm``, ``alpha``, ``kappa`` and ``obj``. The
    function ignores a ``multi_`` prefix (multi-needle NIAH).
    """
    name = name[len("multi_"):] if name.startswith("multi_") else name

    for prefix, (b, g, ld, tau, lk, lv) in CUSTOM_PREFIXES.items():
        if name == prefix or name.startswith(prefix + "-") or name.startswith(prefix + "_"):
            rest = name[len(prefix):]
            model = MODEL_BASES[0][0]  # default Mistral; a model suffix (e.g. '-llama')
            for base, short in MODEL_BASES[1:]:  # would sit right after the prefix
                if short and rest.startswith(short):
                    model, rest = base, rest[len(short):]
                    break
            _, _, tail = rest.partition("_")
            p = _pairs(tail)
            out = {"mode": "causal_lookahead", "model": model,
                   "cell": cell_name(b, g, ld), "brule": budget_rule_name(tau),
                   "sel": selection_name(lk, lv),
                   "b": b, "g": g, "lD": ld, "tau": tau, "lK": lk, "lV": lv,
                   "budget": p.get("B"), "chunk": p.get("chunk")}
            for k in ("M", "T", "lL", "lT", "nrm", "alpha", "kappa"):
                out[k] = p.get(k)
            out["obj"] = p.get("obj", "projected")
            return out

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

    m = _BASELINE_RE.match(name)
    if not m:
        return None
    mode = m.group("mode")
    out = {"mode": mode, "model": m.group("model"),
           "budget": m.group("budget"), "chunk": m.group("chunk")}
    if mode == "criticalkv":
        out["mode"] = "critical"
    elif mode == "critical":                 # Ada-KV run under the old mode name
        out["mode"], out["legacy_critical"] = "ada", True
    out.update(_pairs(m.group("tail")))
    return out


def main():
    """Print the folder name of the run that the arguments specify."""
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
    ap.add_argument("--budget-tau")
    ap.add_argument("--score-objective")
    ap.add_argument("--name-prefix", help="Override the identity string (RUN_NAME_PREFIX); "
                     "see CUSTOM_PREFIXES for how it round-trips through parse_run_name.")
    a = ap.parse_args()
    print(build_run_name(
        mode=a.mode, model_tag=a.model_tag, budget=a.budget, chunk_size=a.chunk_size,
        num_particles=a.num_particles, lookahead_len=a.lookahead_len,
        lambda_lookahead=a.lambda_lookahead, lookahead_temperature=a.lookahead_temperature,
        beta=a.beta, gamma=a.gamma, lambda_d=a.lambda_d,
        normalize_typicality=a.normalize_typicality,
        robust_alpha=a.robust_alpha, robust_topk_beta=a.robust_topk_beta,
        lambda_k=a.lambda_k, lambda_v=a.lambda_v, budget_tau=a.budget_tau,
        score_objective=a.score_objective,
        name_prefix=a.name_prefix))


if __name__ == "__main__":
    main()
