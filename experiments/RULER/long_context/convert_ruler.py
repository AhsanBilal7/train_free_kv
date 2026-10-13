#!/usr/bin/env python
"""Convert the output of the NVIDIA RULER generator to the schema of simonjegou/ruler.

simonjegou/ruler contains only the 4096, 8192 and 16384 configs. make_data.sh runs
the NVIDIA generator for longer configs, and this script converts its output to
the same columns::

    context, question, answer_prefix, answer (list[str]), task, max_new_tokens

``prepare.py --model_template_type base`` writes one jsonl file per task. Its
``input`` field is <context><question><answer_prefix> as one string. The Hub
export divides this string at fixed markers, and this script does the same
(validate.py compares the rows with the Hub 4096 config):

    niah_*      The question starts at the last "\nWhat ". The answer prefix
                starts at the last " The special magic".
    vt/cwe/fwe  The question starts at the last "Question: ".
    qa_*        The question starts at the last "Answer the question based on
                the given documents."
    others      The answer prefix starts at the last " Answer:".

The space between the question and the answer prefix stays with the question, as
in the Hub rows.

Output: ``<out_dir>/<length>/test-00000-of-00001.parquet`` and a README with the
config list, so that ``load_dataset(<out_dir>, "<length>")`` finds each length.
"""
import argparse
import json
import os

import pandas as pd

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]

# RULER scripts/synthetic.yaml -> constants.py `tokens_to_generate`, identical to
# the max_new_tokens column of the Hub configs.
MAX_NEW_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}


def _family(task):
    return task.split("_")[0]


def split_input(task, text):
    fam = _family(task)
    if fam == "niah":
        q = text.rindex("\nWhat ") + 1
        a = text.rindex(" The special magic") + 1
    elif fam == "qa":
        q = text.rindex("Answer the question based on the given documents.")
        a = text.rindex(" Answer:") + 1
    else:
        q = text.rindex("Question: ")
        a = text.rindex(" Answer:") + 1
    if not 0 < q < a <= len(text):
        raise ValueError(f"{task}: bad split q={q} a={a} len={len(text)}")
    return text[:q], text[q:a], text[a:]


def load_task(raw_dir, task):
    path = os.path.join(raw_dir, task, "validation.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ctx, question, prefix = split_input(task, r["input"])
            rows.append({
                "context": ctx,
                "question": question,
                "answer_prefix": prefix,
                "answer": list(r["outputs"]),
                "task": task,
                "max_new_tokens": MAX_NEW_TOKENS[_family(task)],
            })
    return rows


def interleave(per_task):
    """Order the rows round-robin across the tasks: row 0 of each task, then row 1, ...

    pred.py continues a run by row position. With this order each prefix of a run
    contains the same number of rows from each task, so an interim score
    (progress.py) is a balanced preview. The final scores do not depend on the order.
    """
    out = []
    for i in range(max((len(r) for r in per_task), default=0)):
        out += [rows[i] for rows in per_task if i < len(rows)]
    return out


def write_readme(out_dir):
    """Write a README with the config list, so that ``load_dataset(out_dir, "<length>")`` finds each length."""
    lengths = sorted((d for d in os.listdir(out_dir)
                      if d.isdigit() and os.path.isdir(os.path.join(out_dir, d))), key=int)
    lines = ["---", "configs:"]
    for L in lengths:
        lines += [f"- config_name: '{L}'", "  data_files:",
                  "  - split: test", f"    path: {L}/test-*"]
    lines += ["---", "",
              "RULER long-context configs generated locally with NVIDIA RULER",
              "(see experiments/RULER/long_context/make_data.sh), in the",
              "simonjegou/ruler schema.", ""]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="RULER prepare.py --save_dir for one length")
    ap.add_argument("--length", required=True, type=int)
    ap.add_argument("--out_dir", required=True, help="local dataset root (holds <length>/ dirs)")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--allow_partial", action="store_true",
                    help="write even if some tasks are missing (validation only)")
    args = ap.parse_args()

    per_task = []
    for task in args.tasks:
        path = os.path.join(args.raw_dir, task, "validation.jsonl")
        if not os.path.exists(path):
            if args.allow_partial:
                print(f"skip missing {task}")
                continue
            raise FileNotFoundError(path)
        task_rows = load_task(args.raw_dir, task)
        print(f"{task}: {len(task_rows)} rows")
        per_task.append(task_rows)
    rows = interleave(per_task)

    os.makedirs(os.path.join(args.out_dir, str(args.length)), exist_ok=True)
    out = os.path.join(args.out_dir, str(args.length), "test-00000-of-00001.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    write_readme(args.out_dir)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
