#!/usr/bin/env python
"""Write the LaTeX body of a LongBench results table from ``pred/``.

The per-backbone tables of the paper have 16 datasets, 4 methods and 4 budgets,
and a Full Cache row. The script reads the predictions, scores them with the
scorer of ``experiments/LongBench/eval.py`` and prints the rows.

The script scores the ``.jsonl`` files and does not read ``result.json``.
``eval.py`` writes ``result.json`` only at the end of a sweep.

A cell without predictions shows \\tbd. A row average shows \\tbd if one of the 16
datasets is missing.

Usage::

    python scripts/longbench_table.py --backbone qwen2
    python scripts/longbench_table.py --backbone ministral3 --budgets 128 256
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONGBENCH_DIR = os.path.join(REPO_ROOT, "experiments", "LongBench")
sys.path.insert(0, LONGBENCH_DIR)                    # eval.py and metrics.py
sys.path.append(REPO_ROOT)

# Column order and headers exactly as the paper's tables have them.
DATASETS = [
    ("narrativeqa", "NrtvQA"), ("qasper", "Qasper"), ("multifieldqa_en", "MF-en"),
    ("hotpotqa", "HotpotQA"), ("2wikimqa", "2WikiMQA"), ("musique", "Musique"),
    ("gov_report", "GovReport"), ("qmsum", "QMSum"), ("multi_news", "MultiNews"),
    ("trec", "TREC"), ("triviaqa", "TriviaQA"), ("samsum", "SAMSum"),
    ("passage_count", "PCount"), ("passage_retrieval_en", "PRe"),
    ("lcc", "Lcc"), ("repobench-p", "RB-P"),
]

# backbone key -> (MODEL_TAG used in folder names, run_naming suffix)
BACKBONES = {
    "mistral":    ("Mistral-7B-Instruct-v0.2", ""),
    "llama":      ("Llama-3.1-8B-Instruct", "-llama"),
    "qwen2":      ("Qwen2.5-14B-Instruct", "-qwen2"),
    "ministral3": ("Ministral-3-8B-Instruct-2512", "-ministral3"),
    "gemma4":     ("gemma-4-12B-it", "-gemma4"),
    "qwen35":     ("Qwen3.5-9B", "-qwen35"),
}

# The lookahead cells differ only in reliability coefficients (Table 3 of the paper).
LOOKAHEAD_CELLS = {"ent": ("1.0", "1.3", "0.0"), "dis": ("1.0", "1.0", "0.3")}


def lookahead_dir(cell, model_tag, budget):
    """Return the run-folder name of LORE-KV /E or /D for one backbone and budget."""
    from scripts.run_naming import build_run_name
    beta, gamma, lam_d = LOOKAHEAD_CELLS[cell]
    return build_run_name(
        mode="causal_lookahead", model_tag=model_tag, budget=budget, chunk_size=4,
        num_particles=4, lookahead_len=8, lambda_lookahead="1.0",
        lookahead_temperature="1.0", beta=beta, gamma=gamma, lambda_d=lam_d,
        normalize_typicality="0", robust_alpha="0.0", robust_topk_beta="1.0",
        lambda_k="0.0", lambda_v="0.0", budget_tau="0.0", score_objective="projected",
    )


def score_dir(pred_root, folder):
    """Return ``{dataset: score}`` for the finished datasets of a run, or ``{}``.

    ``folder`` can be a list of possible names. The first name that exists is used.
    """
    from eval import scorer, dataset2metric
    candidates = [folder] if isinstance(folder, str) else list(folder)
    roots = pred_root if isinstance(pred_root, (list, tuple)) else [pred_root]
    path = next((os.path.join(r, c) for c in candidates for r in roots
                 if os.path.isdir(os.path.join(r, c))), None)
    if path is None:
        return {}
    out = {}
    for ds, _ in DATASETS:
        f = os.path.join(path, f"{ds}.jsonl")
        if not os.path.exists(f) or ds not in dataset2metric:
            continue
        preds, answers, all_classes = [], [], None
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                preds.append(o["pred"]); answers.append(o["answers"])
                all_classes = o.get("all_classes")
        if preds:
            out[ds] = scorer(ds, preds, answers, all_classes)[0]
    return out


def fmt(scores):
    """Return the cells of a row and the row average, without highlighting."""
    cells = [f"{scores[ds]:.2f}" if ds in scores else r"\tbd" for ds, _ in DATASETS]
    complete = len(scores) == len(DATASETS)
    avg = f"{sum(scores.values()) / len(DATASETS):.2f}" if complete else r"\tbd"
    return cells, avg


def row(label, scores, cells=None, avg=None):
    """Return one LaTeX table row."""
    if cells is None:
        cells, avg = fmt(scores)
    return f"{label} & " + " & ".join(cells) + f" & {avg} \\\\"


def highlight(block):
    """Mark the best and second-best cell of each column with \\best and \\second.

    ``block`` is ``[(label, scores)]`` for the method rows of one budget. Full Cache is
    not included. A column with less than two values is not marked.
    """
    out = [list(fmt(s)) for _, s in block]
    for col, (ds, _) in enumerate(DATASETS):
        have = [(i, s[ds]) for i, (_, s) in enumerate(block) if ds in s]
        if len(have) < 2:
            continue
        best_v = max(v for _, v in have)
        # Every row tied at the top is marked best -- marking only one of two
        # equal values and leaving the other plain reads as an error, and exact
        # ties are common here (PCount and PRe are coarse-grained metrics).
        for i, v in have:
            if v == best_v:
                out[i][0][col] = r"\best{%s}" % out[i][0][col]
        runners = [v for _, v in have if v < best_v]
        if runners:
            second_v = max(runners)
            for i, v in have:
                if v == second_v:
                    out[i][0][col] = r"\second{%s}" % out[i][0][col]
    return [row(label, None, cells, avg) for (label, _), (cells, avg) in zip(block, out)]


def emit(backbone, budgets, pred_root, no_highlight=False):
    """Return the table body, from \\toprule to \\bottomrule, as a list of lines."""
    model_tag, _ = BACKBONES[backbone]
    lines = [r"\toprule",
             "Method & " + " & ".join(h for _, h in DATASETS) + r" & Avg. \\",
             r"\midrule",
             row("Full Cache", score_dir(pred_root,
                 [f"full_{model_tag}_128_chunk_4", f"full_{model_tag}_128_chunk_4_red_0.0",
                  f"test_{model_tag}_128_chunk_4_red_0.0"]))]
    for b in budgets:
        lines += ["", r"\midrule",
                  r"\multicolumn{18}{c}{\textit{Budget} $=%d$} \\" % b, r"\midrule"]
        block = [(label, score_dir(pred_root, folder)) for label, folder in (
            ("CriticalKV", [f"critical_{model_tag}_{b}_chunk_4",
                            f"critical_{model_tag}_{b}_chunk_4_red_0.0"]),
            ("AnDPro",     [f"andpro_{model_tag}_{b}_chunk_4",
                            f"andpro_{model_tag}_{b}_chunk_4_red_0.0"]),
            (r"\lorekvE",  lookahead_dir("ent", model_tag, b)),
            (r"\lorekvD",  lookahead_dir("dis", model_tag, b)),
        )]
        lines += highlight(block) if not no_highlight else [row(l, s) for l, s in block]
    lines.append(r"\bottomrule")
    return lines


def inject(tex_path, backbone, body):
    """Replace the table body (\\toprule to \\bottomrule) of ``tab:longbench_<backbone>`` in a .tex file."""
    # A literal search, not a regex: the label contains \l and { }, which as a
    # pattern raise "bad escape \l" -- a failure that is easy to lose to a
    # redirected stderr and looks exactly like "nothing needed updating".
    label = f"\\label{{tab:longbench_{backbone}}}"
    t = open(tex_path).read()
    at = t.find(label)
    if at < 0:
        raise SystemExit(f"{tex_path}: no {label}")
    start = t.index(r"\toprule", at + len(label))
    end = t.index(r"\bottomrule", start) + len(r"\bottomrule")
    new = t[:start] + "\n".join(body) + t[end:]
    if new == t:
        print(f"  {backbone}: unchanged")
        return
    open(tex_path, "w").write(new)
    print(f"  {backbone}: table body updated in {tex_path}")


def main():
    """Print the table body or write it into a .tex file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512, 1024])
    ap.add_argument("--no-highlight", action="store_true",
                    help="skip the \\best/\\second per-column markup")
    ap.add_argument("--inject", metavar="TEX",
                    help="rewrite the matching table body in this .tex file instead of printing")
    ap.add_argument("--pred-root", nargs="+", default=[os.path.join(LONGBENCH_DIR, "pred")],
                    help="run-folder roots, searched in order (e.g. add an archive location)")
    args = ap.parse_args()

    body = emit(args.backbone, args.budgets, args.pred_root, args.no_highlight)
    if args.inject:
        inject(args.inject, args.backbone, body)
        return
    print("\n".join(body))


if __name__ == "__main__":
    main()
