#!/usr/bin/env python
"""Report the prompt lengths that pred.py gives to the model, for each config.

pred.py applies the chat template to context + question and adds the answer
prefix. RULER sets the length of each sample without the chat template, so the
real prompt is longer by some tokens. Use this report to make sure that
MAX_LENGTH is larger than each prompt. A truncated prompt can lose the needle.

Usage::

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
