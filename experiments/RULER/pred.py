r"""RULER prediction driver (paper Sec. 4.3, Figure 3, Tables 7, 10, 15).

The driver runs the 13 RULER tasks (Hsieh et al., 2024;
https://github.com/NVIDIA/RULER) at the selected sequence lengths with one
eviction method. It writes ``pred/<out_name>/<length>.jsonl``. ``eval.py`` scores
the rows with ``string_match_all`` and ``string_match_part``
(``calculate_metrics.py``).

Each row of the Hub dataset ``simonjegou/ruler`` (the kvpress release of RULER)
contains ``context``, ``question``, ``answer_prefix``, the reference ``answer``
list, ``task`` and ``max_new_tokens``. The prompt is the chat template of the
backbone applied to ``context + question``, followed by ``answer_prefix``.
``long_context/make_data.sh`` makes the 64K and 128K configs.

The driver continues an interrupted run from the last complete row of
``<length>.jsonl_tmp``. It removes an incomplete last line. ``--shard k/K``
divides one length across K workers (see :func:`merge_shards`).

Usage (from this directory; ``scripts/sweep.py`` runs it for each setting)::

    python pred.py -m meta-llama/Llama-3.1-8B-Instruct --max_length 140000 \
        -d simonjegou/ruler --mode causal_lookahead --budget 128 \
        --out_name lorekv-ent-unif-topk-llama_B_128 --dataset_name_list 4096 8192
"""
import argparse
import gc
import json
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lorekv.data import load_ruler_config  # noqa: E402
from lorekv.harness import (add_method_args, configure_model, install_method,  # noqa: E402
                            load_model_and_tokenizer, seed_everything)
from lorekv.monkeypatch.lookahead_runner import generate_with_lookahead  # noqa: E402


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True,
                        help="prompt-length cap; must exceed the longest RULER prompt "
                             "(long_context/check_lengths.py sizes it)")
    parser.add_argument("-d", "--dataset", type=str, default="simonjegou/ruler")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--dataset_name_list", type=str, nargs="+", default=None,
                        help="sequence-length configs, e.g. 4096 8192 16384 65536 131072")
    parser.add_argument("--shard", default=os.environ.get("RULER_SHARD") or None, type=str,
                        help="k/K: decode only rows i >= shard_base with (i - shard_base) %% K == k")
    parser.add_argument("--shard_base", default=int(os.environ.get("RULER_SHARD_BASE", "0")),
                        type=int, help="rows < shard_base were already decoded unsharded")
    parser.add_argument("--local_dataset", default=os.environ.get("RULER_LOCAL_DIR"), type=str,
                        help="directory of locally generated RULER configs (64K/128K)")
    add_method_args(parser)
    return parser.parse_args(args)


def build_ruler_input(tokenizer, context, question, answer_prefix, max_length, device):
    """Make the token IDs of one RULER prompt.

    The user turn is ``context + question``. The assistant turn starts with
    ``answer_prefix``, for example "The special magic number for ... is". If the
    tokenizer has a chat template, the function applies it. Otherwise it
    concatenates the text. Middle truncation applies only if the prompt is longer
    than ``max_length``. Set ``max_length`` larger than the longest prompt, because a
    truncation can remove the needle.

    Returns:
        A tuple ``(input_ids, attention_mask)`` on ``device``.
    """
    user_content = context + question
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            return_tensors="pt",
            # Inert for Qwen2.5, Mistral and Llama, whose templates do not read
            # the variable; passed so a hybrid-reasoning Qwen checkpoint cannot
            # silently open a <think> block and consume the whole answer budget.
            enable_thinking=False,
        )
        if answer_prefix:
            prefix_ids = tokenizer(answer_prefix, add_special_tokens=False, return_tensors="pt").input_ids
            input_ids = torch.cat([input_ids, prefix_ids], dim=1)
    else:
        prompt = user_content + answer_prefix
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids

    if input_ids.shape[1] > max_length:
        # Middle truncation can remove the needle. Increase MAX_LENGTH if this occurs
        # (long_context/check_lengths.py gives the required value).
        print(f"WARNING: prompt of {input_ids.shape[1]} tokens truncated to max_length={max_length}")
        half = int(max_length / 2)
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

@torch.no_grad()
def get_pred(model, tokenizer, data, max_length, out_path, mode, tmp_path=None, device="cuda:0"):
    """Decode ``data`` row by row into ``out_path`` and continue from ``tmp_path``."""
    tmp_path = tmp_path or f"{out_path}_tmp"

    # Continue an interrupted run. The _tmp file has one JSON row for each finished
    # sample. Use only complete, valid rows. Decode again from the first incomplete row.
    preds = []
    times = []
    done = 0
    if os.path.exists(tmp_path):
        # Find the end of the last valid row and cut the file there. A new row after
        # an incomplete line would be on the same line and could not be read.
        good_end = 0
        with open(tmp_path, "rb") as f:
            for raw in f:
                if not raw.endswith(b"\n"):
                    break          # unterminated tail = torn write
                line = raw.strip()
                if line:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        break      # torn/garbled line -> resume from here
                    if done >= len(data):
                        break
                    preds.append(obj["pred"])
                    times.append(obj["time"])
                    done += 1
                good_end += len(raw)
        if good_end < os.path.getsize(tmp_path):
            print(f"{tmp_path}: dropping {os.path.getsize(tmp_path) - good_end} bytes of torn tail")
            with open(tmp_path, "r+b") as f:
                f.truncate(good_end)
        if done:
            print(f"{tmp_path}: resuming from {done}/{len(data)} completed rows")

    # Append, so that the finished rows stay in the file.
    with open(tmp_path, "a", encoding="utf-8") as f:
        for json_obj in tqdm(data[done:]):
            input_ids, attention_mask = build_ruler_input(
                tokenizer, json_obj["context"], json_obj["question"], json_obj["answer_prefix"], max_length, device
            )
            context_length = input_ids.shape[-1]
            max_gen = int(json_obj["max_new_tokens"])

            torch.cuda.synchronize()
            t = time.time()
            if mode == "causal_lookahead":
                # custom multi-phase generation (prefill -> lookahead particles -> evict -> decode);
                # see lorekv.monkeypatch.lookahead_runner.generate_with_lookahead.
                eos_ids = [tokenizer.eos_token_id]
                answer_ids = generate_with_lookahead(
                    model, tokenizer, input_ids, attention_mask, max_gen, eos_ids,
                )
                output = torch.cat([input_ids[0], answer_ids], dim=0)
            else:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=[tokenizer.eos_token_id],
                )[0]
            torch.cuda.synchronize()
            t = time.time() - t
            times.append(t)

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            preds.append(pred)

            # One write per row (not json.dump's piecewise writes + a separate "\n"),
            # so a kill leaves at most one unterminated tail for the resume to cut.
            row = {"pred": pred, "answer": list(json_obj["answer"]), "task": json_obj["task"],
                   "length": json_obj["length"], "context_length": context_length, "time": t}
            if "idx" in json_obj:          # shard rows carry their global row index
                row["idx"] = json_obj["idx"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            gc.collect()
            torch.cuda.empty_cache()

    with open(out_path, "w", encoding="utf-8") as f:
        for json_obj, pred, t in zip(data, preds, times):
            row = {"pred": pred, "answer": list(json_obj["answer"]), "task": json_obj["task"],
                   "length": json_obj["length"], "time": t}
            if "idx" in json_obj:
                row["idx"] = json_obj["idx"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Row sharding: run one experiment on several GPUs at the same time.
#
# `--shard k/K --shard_base B` decodes each row i >= B with (i - B) % K == k into
# pred/<run>/<len>.shard{k}of{K}.jsonl_part. Rows < B are the rows that the unsharded
# run finished in <len>.jsonl_tmp. The last shard to finish merges all rows into
# <len>.jsonl, which eval.py reads. eval.py does not read the .jsonl_part files.
# -----------------------------------------------------------------------------
def _read_rows(path, limit=None):
    """Read the complete rows of a jsonl file. Stop at the first incomplete line."""
    rows = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            for raw in f:
                if not raw.endswith(b"\n") or (limit is not None and len(rows) >= limit):
                    break
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError:
                    break
    return rows


def merge_shards(run_dir, length, K, base, n_rows):
    """Write ``<length>.jsonl`` when the unsharded prefix and all K shard files exist.

    Returns:
        True if the file was written, False if shards are missing.
    """
    out_path = os.path.join(run_dir, f"{length}.jsonl")
    parts = [os.path.join(run_dir, f"{length}.shard{k}of{K}.jsonl_part") for k in range(K)]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        print(f"merge: waiting for {len(missing)}/{K} shard(s) of {run_dir}/{length}")
        return False
    prefix = _read_rows(os.path.join(run_dir, f"{length}.jsonl_tmp"), limit=base)
    if len(prefix) < base:
        raise RuntimeError(f"merge: unsharded prefix has {len(prefix)} rows, expected {base}")
    by_idx = {i: r for i, r in enumerate(prefix)}
    for p in parts:
        for r in _read_rows(p):
            by_idx[r["idx"]] = r
    if sorted(by_idx) != list(range(n_rows)):
        raise RuntimeError(f"merge: have {len(by_idx)} distinct rows, expected 0..{n_rows - 1}")
    tmp = out_path + f".merge{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for i in range(n_rows):
            r = by_idx[i]
            f.write(json.dumps({"pred": r["pred"], "answer": r["answer"], "task": r["task"],
                                "length": r["length"], "time": r["time"]}, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)       # atomic: two shards finishing together both write the same file
    print(f"merge: wrote {out_path} ({n_rows} rows from prefix {base} + {K} shards)")
    return True

def main():
    """Parse the arguments, install the method, load the model and decode each length."""
    seed_everything(42)
    args = parse_args()
    lengths = args.dataset_name_list or ["4096", "8192", "16384"]
    print(f"mode={args.mode} budget={args.budget} lengths={lengths}")

    install_method(args.mode)              # must precede model construction
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    configure_model(model, args)

    run_dir = os.path.join("pred", args.out_name)
    os.makedirs(run_dir, exist_ok=True)
    for length in lengths:
        out_path = os.path.join(run_dir, f"{length}.jsonl")
        if os.path.exists(out_path):
            print(f"{out_path} exists, skipping")
            continue
        data = load_ruler_config(args.dataset, args.local_dataset, str(length))
        rows = [{**row, "length": int(length)} for row in data]
        if args.shard:
            k, K = (int(x) for x in args.shard.split("/"))
            part = os.path.join(run_dir, f"{length}.shard{k}of{K}.jsonl_part")
            if not os.path.exists(part):
                mine = [{**rows[i], "idx": i} for i in range(args.shard_base, len(rows))
                        if (i - args.shard_base) % K == k]
                print(f"shard {k}/{K} of {run_dir}/{length}: {len(mine)} rows from {args.shard_base}")
                get_pred(model, tokenizer, mine, args.max_length, part, args.mode,
                         tmp_path=part + "_tmp")
            merge_shards(run_dir, length, K, args.shard_base, len(rows))
            continue
        t = time.time()
        get_pred(model, tokenizer, rows, args.max_length, out_path, args.mode)
        print(f"== {args.out_name} {length} time: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
