"""Score RULER predictions (paper Sec. 4.3, Tables 7 and 10).

For each run folder in ``pred/`` the script scores each ``<length>.jsonl`` with
the string-match metrics of RULER (``calculate_metrics.py``). The two QA tasks use
``string_match_part``. The other eleven tasks use ``string_match_all``, which
gives partial credit over the reference answers (Hsieh et al., 2024; App. E.4).
The script writes ``result.json`` with the score of each task, the 13-task mean
of each length and the mean over the lengths.
"""
import os
import json
import argparse

import pandas as pd

from calculate_metrics import calculate_metrics


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default="pred", help="Directory holding per-run prediction sub-folders")
    return parser.parse_args(args)

def score_length_file(path):
    """Read one ``<length>.jsonl`` and return ``{task: score or None}``."""
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

# The Hub configs; see the note on the overall average below.
STANDARD_LENGTHS = {"4096", "8192", "16384"}


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

        # overall = mean over the per-length averages of the STANDARD lengths only.
        # The paper's RULER "Avg." is over 4K/8K/16K; the 64K/128K configs are added
        # to the same run folders (same method, same hyperparameters), and letting
        # them into this mean would silently change every published average the
        # moment a long-context file lands. They are reported per length instead.
        # A folder with no standard length at all falls back to all its lengths.
        std = [l for l in scores if l in STANDARD_LENGTHS]
        scores["average"] = mean([scores[l]["average"] for l in (std or list(scores))])

        out_path = os.path.join(run_path, "result.json")
        with open(out_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

        print(f"\n=== {run} ===")
        for length in [k for k in scores if k != "average"]:
            print(f"  [{length}] average={scores[length]['average']}")
        print(f"  OVERALL average (4K-16K)={scores['average']}")
        print(f"  wrote {out_path}")
