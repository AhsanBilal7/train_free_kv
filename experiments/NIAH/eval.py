import os
import json
import argparse

import numpy as np

# -----------------------------------------------------------------------------
# NIAH scorer.
#
# Consumes the prediction jsonl files written by pred.py (one per target context
# length under pred/<out_name>/) and scores each (length, depth) cell with the
# gkamradt exact-match rule: 1.0 if the needle answer appears as a
# (case-insensitive, whitespace-trimmed) substring of the model response, else
# 0.0. A softer word-level recall of the answer is also recorded for a smoother
# heatmap when desired (NIAH_METRIC=recall).
#
# For each run we write:
#   * result.json  — the full (length x depth) score grid + per-length means +
#                    an overall mean, plus the word-recall grid.
#   * heatmap.png  — the signature NIAH retrieval heatmap (depth x length).
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
    return parser.parse_args(args)


def _normalize(s):
    return s.strip().lower()


def exact_match(pred, answers):
    """1.0 if any gold answer is a substring of the (normalized) prediction."""
    p = _normalize(pred)
    return 1.0 if any(_normalize(a) in p for a in answers) else 0.0


def word_recall(pred, answers):
    """Fraction of the (best) gold answer's words present in the prediction —
    a graded fallback that produces a smoother heatmap than binary match."""
    p = set(_normalize(pred).split())
    best = 0.0
    for a in answers:
        words = _normalize(a).split()
        if not words:
            continue
        hit = sum(1 for w in words if w in p) / len(words)
        best = max(best, hit)
    return best


def score_length_file(path):
    """Load one <length>.jsonl -> list of {depth, exact_match, recall} dicts."""
    cells = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answers = obj.get("answer") or [obj.get("needle_answer", "")]
            cells.append({
                "depth": float(obj["depth_percent"]),
                "exact_match": exact_match(obj["pred"], answers),
                "recall": word_recall(obj["pred"], answers),
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


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.pred_dir, exist_ok=True)

    for run in sorted(os.listdir(args.pred_dir)):
        run_path = os.path.join(args.pred_dir, run)
        if not os.path.isdir(run_path):
            continue
        length_files = sorted(
            [f for f in os.listdir(run_path) if f.endswith(".jsonl") and not f.endswith("_tmp")],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x,
        )
        if not length_files:
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
        print(f"  OVERALL exact_match={scores['average']['exact_match']} recall={scores['average']['recall']}")
        print(f"  wrote {os.path.join(run_path, 'result.json')}")
