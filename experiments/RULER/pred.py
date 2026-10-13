r"""RULER prediction driver (paper Sec. 4.3, Figure 3, Tables 7, 10, 15).

Runs the 13 RULER tasks (Hsieh et al., 2024; https://github.com/NVIDIA/RULER) at
the requested sequence lengths with one KV-eviction method and writes
``pred/<out_name>/<length>.jsonl``; ``eval.py`` scores them with RULER's
``string_match_all`` / ``string_match_part`` metrics (``calculate_metrics.py``).

Every Hub row (``simonjegou/ruler``, the kvpress release of RULER) ships its own
``context``, ``question``, ``answer_prefix``, gold ``answer`` list, ``task`` and
``max_new_tokens``. The prompt is the backbone's chat template over
``context + question`` with the assistant turn primed by ``answer_prefix``.
64K/128K configs are generated locally (``long_context/make_data.sh``).

Long contexts are expensive, so this driver is resumable at the row level (a torn
last line of ``<length>.jsonl_tmp`` is cut and re-decoded) and can split one
length across several workers with ``--shard k/K`` (see :func:`merge_shards`).

Usage (from this directory; see ``run.sh`` for the sweep wrapper)::

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
    """Assemble a RULER prompt.

    User turn = context + question; the assistant turn is primed with
    `answer_prefix` (e.g. "The special magic number for ... is"). When the
    tokenizer exposes a chat template (Mistral / Llama-Instruct) we apply it so
    the format matches the instruct model's expectations; otherwise we fall back
    to raw concatenation. Truncation (middle) is only a safety net -- with
    max_length >> the largest RULER config it never triggers, so needles are
    preserved.
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
        # A RULER needle can sit anywhere, so middle truncation can delete it and the
        # row then scores as a model failure. This must never fire on a real run --
        # raise MAX_LENGTH instead (long_context/check_lengths.py sizes it).
        print(f"WARNING: prompt of {input_ids.shape[1]} tokens truncated to max_length={max_length}")
        half = int(max_length / 2)
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

@torch.no_grad()
def get_pred(model, tokenizer, data, max_length, out_path, mode, tmp_path=None, device="cuda:0"):
    """Decode ``data`` row by row into ``out_path``, resuming from ``tmp_path``."""
    tmp_path = tmp_path or f"{out_path}_tmp"

    # --- Resume: reuse rows already completed by a prior interrupted run ---
    # The _tmp file streams one JSON object per finished row (flushed each step).
    # Count only fully-flushed (valid-JSON) lines; a truncated final line means the
    # process died mid-write, so we stop there and regenerate that row onward.
    preds = []
    times = []
    done = 0
    if os.path.exists(tmp_path):
        # Read in bytes to track where the last VALID row ends, then cut the file
        # there. Appending after a torn line is not safe: the next row lands on the
        # same line as the fragment, that line never parses, and every later resume
        # stops at it -- so each new allocation re-decodes the same rows forever
        # while the file grows. A wall-clock kill usually lands mid-row, so the torn
        # tail must be removed, not skipped.
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

    # Append so previously-completed rows are preserved (do NOT truncate).
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
# Row sharding: run ONE experiment on several GPUs / memory slots at once.
#
# `--shard k/K --shard_base B` decodes rows B+k, B+k+K, B+k+2K, ..., i.e. every row i >= B with
# (i - B) % K == k, into pred/<run>/<len>.shard{k}of{K}.jsonl_part (resuming from its
# own _tmp, torn-tail safe like the unsharded path). Rows < B are the ones the
# unsharded run already finished in <len>.jsonl_tmp, so sharding an in-flight run
# never redoes work. Whichever shard finishes last merges everything into the
# normal <len>.jsonl (same fields as an unsharded run), which is all eval.py reads.
# The .jsonl_part suffix keeps eval.py's `*.jsonl` scan from reading shards.
# -----------------------------------------------------------------------------
def _read_rows(path, limit=None):
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
    """Write <len>.jsonl once the unsharded prefix and all K shard parts exist."""
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
