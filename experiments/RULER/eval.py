import os
import json
import argparse

import pandas as pd

from calculate_metrics import calculate_metrics

# -----------------------------------------------------------------------------
# RULER scorer.
#
# Consumes the prediction jsonl files written by pred.py (one per sequence-length
# config under pred/<out_name>/) and scores them with the RULER string-match
# metric (calculate_metrics.py: partial match for the `qa` tasks, all-match for
# every other task). For each method run we report per-task scores, the mean over
# the 13 tasks at each length, and the overall mean across lengths.
# -----------------------------------------------------------------------------

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default="pred", help="Directory holding per-run prediction sub-folders")
    return parser.parse_args(args)

def score_length_file(path):
    """Load one <length>.jsonl and return {task: string_match_score_or_None}."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "task": obj["task"],
                "answer": obj["answer"],
                "predicted_answer": obj["pred"],
            })
    df = pd.DataFrame(rows)
    metrics = calculate_metrics(df)  # {task: {"string_match": score} | None}
    return {task: (m["string_match"] if m else None) for task, m in metrics.items()}

def mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None

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

        scores = {}
        for filename in length_files:
            length = filename.split('.')[0]
            per_task = score_length_file(os.path.join(run_path, filename))
            per_task["average"] = mean(per_task.values())
            scores[length] = per_task

        # overall = mean over the per-length averages
        scores["average"] = mean([scores[l]["average"] for l in scores])

        out_path = os.path.join(run_path, "result.json")
        with open(out_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

        print(f"\n=== {run} ===")
        for length in [k for k in scores if k != "average"]:
            print(f"  [{length}] average={scores[length]['average']}")
        print(f"  OVERALL average={scores['average']}")
        print(f"  wrote {out_path}")
