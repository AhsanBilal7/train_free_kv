"""Score Needle-in-a-Haystack predictions (paper Sec. 4.2, Figure 8).

Each cell (length, depth) gets the score of the reference ExactMatchScorer
(https://github.com/gkamradt/LLMTest_NeedleInAHaystack, MIT): the fraction of the
expected fragments that the answer contains (case-insensitive). The score is 1 or
0 for one needle. The script also records a word recall
(``NIAH_METRIC=recall`` uses it as the main metric).

For each run the script writes ``result.json`` (score grid, mean of each length,
overall mean) and ``heatmap.png``. It then prints a summary of all runs. ``--``
shows a length that was not run, and ``tmp`` shows a length that did not finish.
"""
import os
import sys
import csv
import json
import argparse

import numpy as np


# Which score drives result.json's headline means + the heatmap coloring:
#   "exact_match" (default, faithful to gkamradt) or "recall" (graded word recall).
METRIC = os.environ.get("NIAH_METRIC", "exact_match")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default="pred", help="Directory holding per-run prediction sub-folders")
    parser.add_argument('--metric', type=str, default=METRIC, choices=["exact_match", "recall"],
                        help="headline metric for result.json + heatmap")
    parser.add_argument('--no_heatmap', action='store_true', help="skip writing heatmap.png")
    parser.add_argument('--summary_csv', type=str, default=None,
                        help="also write the cross-run summary table to this CSV path")
    return parser.parse_args(args)


def _normalize(s):
    """Normalize a string as ``needlehaystack/scorers/exact_match.py`` does."""
    return s.strip().lower()


def exact_match(pred, fragments):
    """Return the fraction of expected fragments that the answer contains (ExactMatchScorer).

    The comparison is case-insensitive and ignores leading and trailing whitespace.
    For one needle the result is 1.0 or 0.0. For several needles it is the fraction,
    for example 0.667 for 2 of 3 fragments.
    """
    if not fragments:
        # Defensive: nothing to score against -> treat as miss (reference does the same).
        return 0.0
    haystack = _normalize(pred)
    matches = sum(1 for f in fragments if _normalize(f) in haystack)
    return matches / len(fragments)


def word_recall(pred, fragments):
    """Return the fraction of the words of the best reference fragment in the answer.

    This metric is not part of the reference implementation. Set
    ``NIAH_METRIC=recall`` to use it.
    """
    p = set(_normalize(pred).split())
    best = 0.0
    for a in fragments:
        words = _normalize(a).split()
        if not words:
            continue
        hit = sum(1 for w in words if w in p) / len(words)
        best = max(best, hit)
    return best


def row_fragments(obj):
    """Return the reference fragments of one prediction row.

    The function reads ``expected_fragments``. It also reads the older fields
    ``answer`` and ``needle_answer``, so that older files give the same scores.
    """
    frags = obj.get("expected_fragments") or obj.get("answer")
    if not frags:
        single = obj.get("needle_answer")
        frags = [single] if single else []
    return [f for f in frags if f]


def score_length_file(path):
    """Read one ``<length>.jsonl`` and return a list of ``{depth, exact_match, recall}``."""
    cells = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fragments = row_fragments(obj)
            cells.append({
                "depth": float(obj["depth_percent"]),
                "exact_match": exact_match(obj["pred"], fragments),
                "recall": word_recall(obj["pred"], fragments),
                "n_fragments": len(fragments),
            })
    return cells


def mean(values):
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def save_heatmap(grid, lengths, depths, out_path, metric, run_name):
    """Save the score grid as a heatmap. Rows are depths (0% at the top); columns are lengths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("niah", ["#F1746A", "#F6D55C", "#3CB371"])
    arr = np.array(grid, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(lengths) + 3), max(4, 0.5 * len(depths) + 2)))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f"{L}" for L in lengths], rotation=45, ha="right")
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{d:g}%" for d in depths])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Needle depth")
    ax.set_title(f"NIAH retrieval ({metric}) — {run_name}")

    for i in range(len(depths)):
        for j in range(len(lengths)):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Run folders are named and parsed by scripts/run_naming.py.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")))
from run_naming import parse_run_name  # noqa: E402


def split_run_name(run):
    """Return ``(label, model_tag, budget)`` for a run folder, or ``("?", "?", None)``.

    The label is ``lorekv-<cell>-<brule>-<sel>`` for a LORE-KV run and the mode name
    for a baseline.
    """
    p = parse_run_name(run)
    if not p or p.get("budget") is None:
        return "?", "?", None
    if p["mode"] == "causal_lookahead" and p.get("cell"):
        label = f"lorekv-{p['cell']}-{p['brule']}-{p['sel']}"
    else:
        label = p["mode"]
    try:
        budget = int(p["budget"])
    except (TypeError, ValueError):
        return label, p.get("model", "?"), None
    return label, p.get("model", "?"), budget


def print_summary_table(summary, metric, csv_path=None):
    """Print one row for each run and one column for each target length.

    Each cell is the mean over the depths of that length. ``--`` shows a length that
    was not run, and ``tmp`` shows a length that did not finish. OVERALL is the mean
    over all scored cells and N is the number of cells. Runs with a different N are
    not comparable. The rows are grouped by model.
    """
    if not summary:
        return

    all_lengths = sorted({L for row in summary for L in row["all_lengths"]})
    idx_w = max(3, len(str(len(summary))))
    col_w = max([7] + [len(str(L)) + 2 for L in all_lengths])

    header = (f"{'#':>{idx_w}}  " + "".join(f"{L:>{col_w}}" for L in all_lengths)
              + f"{'OVERALL':>10}{'N':>6}  {'COVER':<8}")
    width = len(header) + 3
    print("\n" + "=" * width)
    print(f" NIAH SUMMARY — headline metric: {metric}")
    print(" '--' = length never run   'tmp' = crashed mid-length (only .jsonl_tmp, excluded from the mean)")
    print(" N = scored (length x depth) cells; rows with different N are NOT comparable head-to-head.")
    print("=" * width)

    csv_rows = [["model", "mode", "budget", "run",
                 *[str(L) for L in all_lengths], "overall", "n_cells", "coverage"]]

    # Group by model, then order within a group by mode/budget so the same
    # configuration across methods lines up vertically.
    models = sorted({split_run_name(r["run"])[1] for r in summary})
    i = 0
    n_partial_total = 0
    for model in models:
        rows = [r for r in summary if split_run_name(r["run"])[1] == model]
        rows.sort(key=lambda r: (split_run_name(r["run"])[2] or 0, split_run_name(r["run"])[0], r["run"]))

        # Reference grid for THIS model: the most complete coverage anything reached.
        full = max((set(r["scored_lengths"]) for r in rows), key=len, default=set())
        max_cells = max((r["n_cells"] for r in rows), default=0)

        print(f"\n--- model: {model}   (reference grid: {sorted(full)} = {max_cells} cells) ---")
        print(header + "RUN")
        for row in rows:
            i += 1
            cells = []
            for L in all_lengths:
                v = row["per_length"].get(L)
                if v is not None:
                    cells.append(f"{v:>{col_w}.4f}")
                elif L in row["tmp_lengths"]:
                    cells.append(f"{'tmp':>{col_w}}")
                else:
                    cells.append(f"{'--':>{col_w}}")
            overall = f"{row['overall']:.4f}" if row["overall"] is not None else "--"
            partial = set(row["scored_lengths"]) != full or row["n_cells"] != max_cells
            n_partial_total += partial
            mode, _, budget = split_run_name(row["run"])
            # Coverage marker sits BEFORE the (very long) run name so it stays on screen.
            cover = "PARTIAL" if partial else "full"
            print(f"{i:>{idx_w}}  " + "".join(cells) + f"{overall:>10}{row['n_cells']:>6}  "
                  f"{cover:<8}{row['run']}")
            csv_rows.append([
                model, mode, budget, row["run"],
                *[("" if row["per_length"].get(L) is None else f"{row['per_length'][L]:.4f}")
                  for L in all_lengths],
                overall, str(row["n_cells"]), "partial" if partial else "full",
            ])

    print("\n" + "=" * width)
    if n_partial_total:
        print(f" [warn] {n_partial_total}/{len(summary)} runs are PARTIAL — their OVERALL is averaged over")
        print("        fewer cells than the reference grid for their model. Re-run the missing lengths")
        print("        (delete stale *.jsonl_tmp first) before comparing methods on OVERALL.")
    else:
        print(f" All {len(summary)} runs cover their model's full grid — OVERALL is comparable.")
    print("=" * width)

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(csv_rows)
        print(f" wrote {csv_path}")


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.pred_dir, exist_ok=True)

    summary = []

    for run in sorted(os.listdir(args.pred_dir)):
        run_path = os.path.join(args.pred_dir, run)
        if not os.path.isdir(run_path):
            continue
        length_files = sorted(
            [f for f in os.listdir(run_path) if f.endswith(".jsonl") and not f.endswith("_tmp")],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x,
        )
        # Lengths that started but never finished: pred.py only writes the final
        # <length>.jsonl after the whole depth loop completes, so a leftover
        # <length>.jsonl_tmp means that length crashed (OOM, walltime, ...).
        tmp_lengths = sorted({
            int(f.split('.')[0]) for f in os.listdir(run_path)
            if f.endswith(".jsonl_tmp") and f.split('.')[0].isdigit()
        } - {int(f.split('.')[0]) for f in length_files})

        if not length_files:
            # Nothing scoreable, but still surface it in the table so a run that
            # crashed on its very first length is not silently invisible.
            summary.append({"run": run, "per_length": {}, "tmp_lengths": tmp_lengths,
                            "scored_lengths": [], "all_lengths": tmp_lengths,
                            "overall": None, "n_cells": 0})
            continue

        lengths = [int(f.split('.')[0]) for f in length_files]
        # union of depths seen across all length files, sorted ascending (0% -> 100%)
        per_length_cells = {int(f.split('.')[0]): score_length_file(os.path.join(run_path, f)) for f in length_files}
        depths = sorted({c["depth"] for cells in per_length_cells.values() for c in cells})

        # Build the (depth x length) grids for both metrics.
        em_grid, rc_grid = [], []
        for d in depths:
            em_row, rc_row = [], []
            for L in lengths:
                cell = next((c for c in per_length_cells[L] if c["depth"] == d), None)
                em_row.append(cell["exact_match"] if cell else np.nan)
                rc_row.append(cell["recall"] if cell else np.nan)
            em_grid.append(em_row)
            rc_grid.append(rc_row)

        headline = em_grid if args.metric == "exact_match" else rc_grid

        # result.json: per-length means + overall mean for both metrics + the grid.
        scores = {"metric": args.metric, "lengths": lengths, "depths": depths}
        per_length = {}
        for j, L in enumerate(lengths):
            col_em = [em_grid[i][j] for i in range(len(depths))]
            col_rc = [rc_grid[i][j] for i in range(len(depths))]
            per_length[str(L)] = {
                "exact_match": mean([v for v in col_em if not np.isnan(v)]),
                "recall": mean([v for v in col_rc if not np.isnan(v)]),
            }
        scores["per_length"] = per_length
        flat_em = [v for row in em_grid for v in row if not np.isnan(v)]
        flat_rc = [v for row in rc_grid for v in row if not np.isnan(v)]
        scores["average"] = {"exact_match": mean(flat_em), "recall": mean(flat_rc)}
        scores["grid"] = {"exact_match": em_grid, "recall": rc_grid}

        with open(os.path.join(run_path, "result.json"), "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

        if not args.no_heatmap:
            try:
                save_heatmap(headline, lengths, depths,
                             os.path.join(run_path, "heatmap.png"), args.metric, run)
            except Exception as e:
                print(f"  [warn] heatmap failed for {run}: {e}")

        print(f"\n=== {run} ===")
        for L in lengths:
            print(f"  [{L}] exact_match={per_length[str(L)]['exact_match']} recall={per_length[str(L)]['recall']}")
        if tmp_lengths:
            print(f"  [warn] incomplete (only .jsonl_tmp, excluded from the means): {tmp_lengths}")
        print(f"  OVERALL exact_match={scores['average']['exact_match']} recall={scores['average']['recall']}")
        print(f"  wrote {os.path.join(run_path, 'result.json')}")

        summary.append({
            "run": run,
            "per_length": {L: per_length[str(L)][args.metric] for L in lengths},
            "tmp_lengths": tmp_lengths,
            "scored_lengths": lengths,
            "all_lengths": sorted(set(lengths) | set(tmp_lengths)),
            "overall": scores["average"][args.metric],
            "n_cells": len(flat_em),
        })

    print_summary_table(summary, args.metric, args.summary_csv)
