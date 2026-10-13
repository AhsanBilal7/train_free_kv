import os
import sys
import csv
import json
import argparse

import numpy as np

# -----------------------------------------------------------------------------
# NIAH scorer.
#
# Consumes the prediction jsonl files written by pred.py (one per target context
# length under pred/<out_name>/) and scores each (length, depth) cell with the
# reference ExactMatchScorer (needlehaystack/scorers/exact_match.py): the
# FRACTION of the expected fragments that appear as (case-insensitive,
# whitespace-trimmed) substrings of the model response. Single-needle cells
# carry one fragment, so this is the classic 1.0/0.0 exact match; multi-needle
# cells are graded (2 of 3 ingredients -> 0.667). A softer word-level recall is
# also recorded for a smoother heatmap when desired (NIAH_METRIC=recall).
#
# For each run we write:
#   * result.json  — the full (length x depth) score grid + per-length means +
#                    an overall mean, plus the word-recall grid.
#   * heatmap.png  — the signature NIAH retrieval heatmap (depth x length).
#
# Finally a cross-run SUMMARY TABLE is printed: one row per run, one column per
# target context length (the per-length mean), plus the overall mean and the
# number of scored cells. Because a run is only averaged over the lengths that
# actually produced a final <length>.jsonl, two runs can silently report means
# over different denominators — the table makes that explicit by marking
#   '--'   a length that was never run, and
#   'tmp'  a length that crashed mid-way (only <length>.jsonl_tmp exists, so
#          pred.py never wrote the final file and it is excluded from the mean).
# Rows whose length set differs from the most complete one are flagged so the
# numbers are not compared across unequal grids.
# -----------------------------------------------------------------------------

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
    """needlehaystack/scorers/exact_match.py::_normalize"""
    return s.strip().lower()


def exact_match(pred, fragments):
    """Reference ExactMatchScorer: the FRACTION of expected fragments that appear
    as (case-insensitive, whitespace-trimmed) substrings of the response.

        fragments = needle.metadata["expected_fragments"] or [expected_answer]
        value     = matches / len(fragments)

    For a single-needle cell `fragments` has one entry, so this reduces exactly
    to the classic 1.0 / 0.0 exact match. For a multi-needle cell it is graded:
    2 of 3 pizza ingredients recalled scores 0.667, matching the reference
    rather than an all-or-nothing (or any-of) rule."""
    if not fragments:
        # Defensive: nothing to score against -> treat as miss (reference does the same).
        return 0.0
    haystack = _normalize(pred)
    matches = sum(1 for f in fragments if _normalize(f) in haystack)
    return matches / len(fragments)


def word_recall(pred, fragments):
    """Fraction of the (best) gold fragment's words present in the prediction —
    a graded fallback that produces a smoother heatmap than substring match.
    NOT part of the reference implementation; opt in with NIAH_METRIC=recall."""
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
    """Gold fragments for one prediction row.

    `expected_fragments` is what pred.py writes now. `answer` (a list, mirroring
    RULER) and the older scalar `needle_answer` are accepted so jsonl files
    produced before the multi-needle rewrite still score identically — for a
    single needle all three carry the same one-element list."""
    frags = obj.get("expected_fragments") or obj.get("answer")
    if not frags:
        single = obj.get("needle_answer")
        frags = [single] if single else []
    return [f for f in frags if f]


def score_length_file(path):
    """Load one <length>.jsonl -> list of {depth, exact_match, recall} dicts."""
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
    """grid[d_idx][l_idx] in [0,1]; rows=depths (0% at top), cols=lengths."""
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


# out_name is built by scripts/run_naming.py, which is also what parses it here, so
# the summary table can never disagree with what run.sh wrote. causal_lookahead runs
# are `lorekv-<cell>-<brule>-<sel>[-<model>]_B_<B>_chunk_<cs>_...`; baselines keep
# `<mode>_<MODEL_TAG>_<B>_chunk_<cs>_red_<red>`. Both are handled, as are folders
# written before the reparametrization.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")))
from run_naming import parse_run_name  # noqa: E402


def split_run_name(run):
    """-> (label, model_tag, budget). Falls back to ('?', '?', None) if unparsed.

    `label` is the run's identity: the config-style `lorekv-<cell>-<brule>-<sel>`
    for lookahead runs (readable, and the same string as its .conf file), or the
    plain mode name for a baseline.
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
    """Cross-run table: rows = runs, cols = target context lengths.

    Each cell is that run's mean over the depths at that length; '--' means the
    length was never run and 'tmp' means it crashed mid-way (only a _tmp file
    exists, so it contributes nothing to the mean). OVERALL is the mean over
    every scored cell and N is how many cells that was — runs with different N
    average over different denominators and are NOT comparable.

    Rows are grouped by model, since the reference grid ("how many lengths did
    this sweep actually cover") is only meaningful within one model."""
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
