#!/usr/bin/env python
"""Report the prompt length pred.py will actually feed the model, per config.

pred.py wraps context + question in the chat template and appends the answer
prefix, then middle-truncates anything longer than --max_length. RULER sizes
each sample against the RAW template (no chat wrapper), so the real prompt is
a few dozen tokens longer than RULER's own `length`. This prints the
distribution so MAX_LENGTH can be checked against it: a truncated RULER prompt
can cut the needle out, which scores as a model failure.

    python check_lengths.py --data_dir <dir> --lengths 65536 131072 --tokenizer <path>
"""
import argparse
import os

import pandas as pd
from transformers import AutoTokenizer


def prompt_len(tok, row):
    ids = tok.apply_chat_template([{"role": "user", "content": row["context"] + row["question"]}],
                                  add_generation_prompt=True)
    return len(ids) + len(tok(row["answer_prefix"], add_special_tokens=False).input_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--lengths", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--per_task", type=int, default=20, help="rows sampled per task")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    for L in args.lengths:
        df = pd.read_parquet(os.path.join(args.data_dir, L, "test-00000-of-00001.parquet"))
        print(f"== {L}: {len(df)} rows, {df.task.nunique()} tasks")
        worst = 0
        for task, g in df.groupby("task"):
            n = [prompt_len(tok, r) + int(r["max_new_tokens"]) for _, r in g.head(args.per_task).iterrows()]
            worst = max(worst, max(n))
            print(f"  {task:16s} rows={len(g):4d}  prompt+gen tokens min={min(n)} max={max(n)}")
        print(f"  worst prompt+max_new_tokens = {worst}  (config length {L})")


if __name__ == "__main__":
    main()
