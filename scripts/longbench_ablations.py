#!/usr/bin/env python
"""Write the LaTeX of the appendix ablation tables.

Tables:
    (i) Score objective: projected (default), attention and unprojected, for each
        budget and the 16 datasets.
    (ii) Lookahead geometry: the M x T matrix of averages at B = 128.
    (iii) The Qwen2.5-14B and Ministral-3-8B tables, made with
        ``longbench_table.emit()``. Their labels are ``tab:new_backbone_*``.

The scoring, \\tbd and the \\best / \\second marks come from
``scripts/longbench_table.py``. The scores come from the ``.jsonl`` files.

Usage::

    python scripts/longbench_ablations.py                  # LaTeX to stdout
    python scripts/longbench_ablations.py --out ablations.tex
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "experiments", "LongBench"))   # eval.py
sys.path.append(_REPO)

from scripts.longbench_table import (            # noqa: E402
    BACKBONES, DATASETS, emit, fmt, highlight, row, score_dir,
)
from scripts.run_naming import build_run_name     # noqa: E402

# The reliability setting both ablations hold fixed: LORE-KV/E, i.e. Table 3's
# (beta, gamma, lambda_D) = (1.0, 1.3, 0.0). Varying one axis at a time is the
# whole point, so everything else below matches the main table's \lorekvE cell.
ENT = dict(beta="1.0", gamma="1.3", lambda_d="0.0")

BUDGETS = [128, 256, 512, 1024]
OBJECTIVES = [("projected", "Projected (default)"),
              ("attention", "Attention"),
              ("unprojected", "Unprojected")]
PARTICLES = [0, 1, 2, 4, 8]
LOOKAHEADS = [2, 4, 8, 16, 32]


def run_dir(model_tag, budget, num_particles, lookahead_len, score_objective):
    return build_run_name(
        mode="causal_lookahead", model_tag=model_tag, budget=budget, chunk_size=4,
        num_particles=num_particles, lookahead_len=lookahead_len,
        lambda_lookahead="1.0", lookahead_temperature="1.0",
        normalize_typicality="0", robust_alpha="0.0", robust_topk_beta="1.0",
        lambda_k="0.0", lambda_v="0.0", budget_tau="0.0",
        score_objective=score_objective, **ENT)


def objective_table(pred_root, model_tag, no_highlight=False):
    lines = [r"\toprule",
             "Objective & " + " & ".join(h for _, h in DATASETS) + r" & Avg. \\"]
    for b in BUDGETS:
        lines += [r"\midrule",
                  r"\multicolumn{18}{c}{\textit{Budget} $=%d$} \\" % b, r"\midrule"]
        block = [(label, score_dir(pred_root, run_dir(model_tag, b, 4, 8, obj)))
                 for obj, label in OBJECTIVES]
        lines += [row(l, s) for l, s in block] if no_highlight else highlight(block)
    lines.append(r"\bottomrule")
    return lines


def geometry_table(pred_root, model_tag):
    """Return the M x T matrix of averages at B = 128.

    A cell is the mean over the 16 datasets, or \\tbd if the run does not have all of
    them.
    """
    lines = [r"\toprule",
             r"$M$ & " + " & ".join(r"$T=%d$" % t for t in LOOKAHEADS) + r" \\",
             r"\midrule"]
    for m in PARTICLES:
        cells = []
        for t in LOOKAHEADS:
            _, avg = fmt(score_dir(pred_root, run_dir(model_tag, 128, m, t, "projected")))
            cells.append(avg)
        # M=0 is the single-trajectory limit: with no particles there is no
        # lookahead simulation, so T is unused and every T gives the SAME run. That
        # is not an inference -- the five M=0 prediction sets are byte-identical
        # apart from their `time` field, and score to 35.685 at 10 decimals. Five
        # identical cells would read as five measurements agreeing, which is a much
        # stronger claim than "this row does not depend on the column".
        distinct = {c for c in cells if c != r"\tbd"}
        if m == 0 and len(distinct) == 1:
            lines.append(f"${m}$ & " + r"\multicolumn{%d}{c}{%s\ \ \small(independent of $T$)} \\"
                         % (len(LOOKAHEADS), distinct.pop()))
            continue
        lines.append(f"${m}$ & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    return lines


# New-backbone tables. These use the SAME emit() as the paper's own
# tab:longbench_<backbone>, so a number here and the corresponding number in
# main_v2 come from one code path -- but the labels must differ, or including both
# in one document gives duplicate \label and \ref silently resolving to the wrong
# table. Hence tab:new_backbone_* here vs tab:longbench_* there.
NEW_BACKBONES = [
    ("qwen2", "Qwen2.5-14B-Instruct"),
    ("ministral3", "Ministral-3-8B-Instruct-2512"),
]

BACKBONE_CAPTION = r"""\caption{\textbf{LongBench on %s.} All four cache budgets against the
matched baselines, with the uncompressed Full Cache row for reference. Best and
second-best per column are marked among the eviction methods only; the Full Cache
row is a reference, not a competitor.}"""


def backbone_tables(pred_root, budgets, no_highlight=False):
    out = []
    for key, pretty in NEW_BACKBONES:
        out += [""]
        out += wrap(emit(key, budgets, pred_root, no_highlight),
                    f"tab:new_backbone_{key}",
                    BACKBONE_CAPTION % pretty.replace("_", r"\_"),
                    "l" + "c" * (len(DATASETS) + 1))
    return out


HEADER = r"""% Additional experiments for LORE-KV: appendix ablations.
%
% GENERATED by scripts/longbench_ablations.py from experiments/LongBench/pred/ --
% regenerate rather than editing cells by hand, so the numbers cannot drift out of
% step with the predictions they come from:
%
%   cd experiments/LongBench
%   python ../../scripts/longbench_ablations.py --out ../../paper_files/additional_experiments.tex
%
% \tbd marks a run that has not finished all 16 datasets. A row or cell average is
% \tbd unless every dataset is present, so no average is ever taken over a subset.
%
% Expects the same preamble macros as the main paper: \best, \second, \tbd,
% \lorekvE, \lorekvD.
%
% The Qwen2.5-14B / Ministral-3-8B tables below are the same numbers as the paper's
% tab:longbench_qwen2 / tab:longbench_ministral3 -- same emit(), same pred/ -- but
% are labelled tab:new_backbone_* so that including this file alongside the main
% text does not create a duplicate \label, where \ref resolves to whichever came
% last with no error. Reference whichever copy belongs to the document you are
% building, and regenerate both after new runs land.
"""

OBJ_CAPTION = r"""\caption{\textbf{Perturbation objective (App.~C\,(iii)).} LongBench scores for
\lorekvE\ on \backboneName\ when the token score is computed against the
\emph{projected} residual (the default), raw \emph{attention} weights, or the
\emph{unprojected} value residual. All other settings are held at the main-table
configuration ($M{=}4$, $T{=}8$, $(\beta,\gamma,\lambda_D){=}(1.0,1.3,0.0)$), so
each block isolates the objective.}"""

GEO_CAPTION = r"""\caption{\textbf{Lookahead geometry.} Average LongBench score for \lorekvE\ on
\backboneName\ at $B{=}128$ across particle count $M$ and lookahead length $T$.
$M{=}0$ disables particle averaging and is the single-trajectory limit; larger $M$
averages more sampled continuations per step, larger $T$ looks further ahead.}"""


def wrap(env_lines, label, caption, colspec, small=True):
    out = [r"\begin{table}[t]", r"\centering"]
    if small:
        out.append(r"\footnotesize")
        out.append(r"\setlength{\tabcolsep}{2.5pt}")
    out += [caption, r"\label{%s}" % label,
            r"\begin{tabular}{%s}" % colspec]
    out += env_lines
    out += [r"\end{tabular}", r"\end{table}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mistral", choices=sorted(BACKBONES))
    ap.add_argument("--pred-root", nargs="+",
                    default=[os.path.join(_REPO, "experiments", "LongBench", "pred")])
    ap.add_argument("--no-highlight", action="store_true")
    ap.add_argument("--no-backbones", action="store_true",
                    help="omit the Qwen2.5-14B / Ministral-3-8B tables")
    ap.add_argument("--out", metavar="TEX", help="write the file instead of stdout")
    args = ap.parse_args()

    model_tag, _ = BACKBONES[args.backbone]
    roots = args.pred_root

    body = [HEADER,
            r"\newcommand{\backboneName}{%s}" % model_tag.replace("_", r"\_"),
            ""]
    body += wrap(objective_table(roots, model_tag, args.no_highlight),
                 "tab:ablation_objective", OBJ_CAPTION,
                 "l" + "c" * (len(DATASETS) + 1))
    body += [""]
    body += wrap(geometry_table(roots, model_tag),
                 "tab:ablation_geometry", GEO_CAPTION,
                 "l" + "c" * len(LOOKAHEADS), small=False)
    if not args.no_backbones:
        body += backbone_tables(roots, BUDGETS, args.no_highlight)
    text = "\n".join(body) + "\n"

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
