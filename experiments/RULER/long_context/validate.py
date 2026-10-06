#!/usr/bin/env python
"""Check the local generator + converter against the Hub's own configs.

Generate a few samples per task at a length the Hub already has (4096), convert
them with convert_ruler.py, and require every converted row to be IDENTICAL,
column for column, to a row of the Hub config. If this passes, the 64k/128k
configs built by the same pipeline follow the same protocol as 4k-16k: same
RULER revision, tokenizer, seed, prompt wording and context/question split.

    python validate.py --local <out_dir> --hub <path to simonjegou/ruler snapshot> --length 4096
"""
import argparse
import os
import sys

import pandas as pd

COLS = ["context", "question", "answer_prefix", "answer", "task", "max_new_tokens"]


def _key(row):
    return (row["task"], row["context"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--hub", required=True)
    ap.add_argument("--length", default="4096")
    args = ap.parse_args()

    loc = pd.read_parquet(os.path.join(args.local, args.length, "test-00000-of-00001.parquet"))
    hub = pd.read_parquet(os.path.join(args.hub, args.length, "test-00000-of-00001.parquet"))
    index = {_key(r): r for _, r in hub.iterrows()}

    bad = 0
    for task, grp in loc.groupby("task"):
        ok = 0
        for _, r in grp.iterrows():
            h = index.get(_key(r))
            if h is None:
                print(f"  {task}: context not found in Hub config")
                bad += 1
                continue
            diffs = [c for c in COLS
                     if (list(r[c]) != list(h[c]) if c == "answer" else r[c] != h[c])]
            if diffs:
                bad += 1
                for c in diffs:
                    print(f"  {task}: {c} differs\n    local={r[c]!r:.200}\n    hub  ={h[c]!r:.200}")
            else:
                ok += 1
        print(f"{task:16s} {ok}/{len(grp)} identical to Hub")
    print("PASS" if bad == 0 else f"FAIL ({bad} rows differ)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
