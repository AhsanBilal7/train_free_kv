#!/usr/bin/env python
"""Show the progress and interim scores of long-context RULER runs.

pred.py writes each finished row to ``pred/<run>/<length>.jsonl_tmp``. It writes
``<length>.jsonl`` only after all 6500 rows. The long configs are ordered
round-robin across tasks (see convert_ruler.py), so each prefix of the ``_tmp``
file is a balanced sample. For each run and length the script prints the number
of rows, the rows of each task, the mean time per row and the interim 13-task
average.

Usage::

    cd experiments/RULER && python long_context/progress.py [--lengths 65536 131072] [--filter Llama]
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from calculate_metrics import calculate_metrics  # noqa: E402


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break              # partial trailing line, as pred.py's resume does
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="pred")
    ap.add_argument("--lengths", nargs="+", default=["65536", "131072"])
    ap.add_argument("--filter", default="", help="substring a run folder must contain")
    args = ap.parse_args()

    for run in sorted(os.listdir(args.pred_dir)):
        if args.filter not in run:
            continue
        for L in args.lengths:
            final = os.path.join(args.pred_dir, run, f"{L}.jsonl")
            tmp = final + "_tmp"
            path = final if os.path.exists(final) else tmp
            # Sharded runs (pred.py --shard): the unsharded prefix plus every shard's
            # finished part or in-flight _tmp. Parts are complete; a shard has a _tmp only
            # until it finishes, so taking the part when present never double-counts.
            shards = []
            if path == tmp:
                d = os.path.join(args.pred_dir, run)
                for f in sorted(os.listdir(d)):
                    if f.startswith(f"{L}.shard") and f.endswith(".jsonl_part"):
                        shards.append(os.path.join(d, f))
                    elif f.startswith(f"{L}.shard") and f.endswith(".jsonl_part_tmp") \
                            and not os.path.exists(os.path.join(d, f[:-4])):
                        shards.append(os.path.join(d, f))
            if not os.path.exists(path) and not shards:
                continue
            rows = read_rows(path) if os.path.exists(path) else []
            if shards:
                base = int(open(os.path.join(args.pred_dir, run, f"{L}.shard_base")).read()) \
                    if os.path.exists(os.path.join(args.pred_dir, run, f"{L}.shard_base")) else len(rows)
                rows = rows[:base]
                for sp in shards:
                    rows += read_rows(sp)
            if not rows:
                print(f"{run} [{L}] 0 rows")
                continue
            df = pd.DataFrame({"task": [r["task"] for r in rows],
                               "answer": [r["answer"] for r in rows],
                               "predicted_answer": [r["pred"] for r in rows]})
            per_task = {t: (m["string_match"] if m else None)
                        for t, m in calculate_metrics(df).items()}
            vals = [v for v in per_task.values() if v is not None]
            avg = round(sum(vals) / len(vals), 2) if vals else None
            secs = sum(r["time"] for r in rows) / len(rows)
            counts = df.task.value_counts()
            state = "DONE" if path == final else "running"
            state += f" ({len(shards)} shards)" if shards else ""
            print(f"{run}\n  [{L}] {state}: {len(rows)}/6500 rows "
                  f"({counts.min()}-{counts.max()} per task), {secs:.1f} s/row, "
                  f"interim avg over {len(vals)} tasks = {avg}")
            print("    " + "  ".join(f"{t}={v}" for t, v in sorted(per_task.items())))


if __name__ == "__main__":
    main()
