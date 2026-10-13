r"""📚 LongBench prediction driver (paper Sec. 4.1, Tables 1, 2, 6, 8, 9, 11-14).

Decodes every sample of the requested LongBench sub-tasks with one KV-eviction
method and writes ``pred/<out_name>/<task>.jsonl`` (one ``{"pred", "answers",
"all_classes", "length", "time"}`` row per sample); ``eval.py`` scores them.

Prompting, middle truncation and the per-task generation lengths follow the
official LongBench ``pred.py`` (https://github.com/THUDM/LongBench, MIT) and the
AnDPro evaluation protocol (Geng et al., NeurIPS 2025), so baseline numbers are
comparable with those papers. Generation is greedy for every method; LORE-KV
(``--mode causal_lookahead``) replaces ``model.generate`` with
:func:`lorekv.monkeypatch.lookahead_runner.generate_with_lookahead`, which samples
the lookahead trajectories, evicts once, and then decodes greedily from the
compressed cache.

A task whose output file already exists is skipped, so an interrupted run resumes
where it stopped.

Usage (from this directory; see ``run.sh`` for the sweep wrapper)::

    python pred.py -m mistralai/Mistral-7B-Instruct-v0.2 --max_length 64000 \
        -d THUDM/LongBench --mode causal_lookahead --budget 128 \
        --out_name lorekv-ent-unif-topk_B_128 --dataset_name_list qasper
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
from lorekv.data import LONGBENCH_NO_CHAT, LONGBENCH_TASKS, load_longbench_split  # noqa: E402
from lorekv.harness import (add_method_args, configure_model, install_method,  # noqa: E402
                            load_model_and_tokenizer, seed_everything)
from lorekv.monkeypatch.lookahead_runner import generate_with_lookahead  # noqa: E402


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True,
                        help="prompt-length cap in tokens; longer prompts are middle-truncated")
    parser.add_argument("-d", "--dataset", type=str, default="THUDM/LongBench",
                        help="Hub id or a local directory of LongBench jsonl files")
    parser.add_argument("--out_name", type=str, required=True,
                        help="run folder under pred/ (scripts/run_naming.py builds it)")
    parser.add_argument("--dataset_name_list", type=str, nargs="+", default=None,
                        help="LongBench sub-tasks to run (default: the 16 English tasks)")
    add_method_args(parser)
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    """Wrap a LongBench prompt in the backbone's instruction format (App. E.1).

    Mistral-7B-Instruct-v0.2 takes the raw prompt, matching Geng et al. (2025).
    Llama-3.1 and Qwen use their chat templates. Ministral-3 uses Mistral's
    ``[INST]...[/INST]`` format rather than its packaged template, which would
    prepend a several-hundred-token system prompt to every long-context input.
    """
    name = model_name.lower()
    if "llama-3" in name and "instruct" in name:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    if "qwen" in name:
        # enable_thinking=False is inert for Qwen2.5 and keeps a hybrid-reasoning
        # Qwen checkpoint from spending the short LongBench budget in a <think> block.
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    if "ministral" in name or "llama2" in name:
        return f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "qwen" in model_name.lower():
        # A stray reasoning block would otherwise count as part of the answer.
        if "</think>" in response:
            response = response.split("</think>")[-1]
        response = response.strip()
    return response


@torch.no_grad()
def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset,
             model_name_or_path, out_path, mode, device="cuda:0"):
    """Decode every row of one sub-task and write ``out_path`` (jsonl)."""
    eos_ids = [tokenizer.eos_token_id]
    if dataset == "samsum":
        # Stop at the first newline: the model otherwise keeps writing "\nDialogue".
        eos_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

    rows = []
    with open(f"{out_path}_tmp", "w", encoding="utf-8") as f:
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # Truncate in the middle: both ends may carry the instructions.
            ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(ids) > max_length:
                half = int(max_length / 2)
                prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                          + tokenizer.decode(ids[-half:], skip_special_tokens=True))
            if dataset not in LONGBENCH_NO_CHAT:
                prompt = build_chat(tokenizer, prompt, model_name_or_path)
            inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = inputs.input_ids.shape[-1]

            torch.cuda.synchronize()
            t = time.time()
            if mode == "causal_lookahead":
                answer_ids = generate_with_lookahead(
                    model, tokenizer, inputs.input_ids, inputs.attention_mask, max_gen, eos_ids)
                output = torch.cat([inputs.input_ids[0], answer_ids], dim=0)
            else:
                extra = {"min_length": context_length + 1} if dataset == "samsum" else {}
                output = model.generate(
                    **inputs, max_new_tokens=max_gen, num_beams=1, do_sample=False,
                    temperature=1.0, eos_token_id=eos_ids, **extra)[0]
            torch.cuda.synchronize()
            t = time.time() - t

            pred = post_process(
                tokenizer.decode(output[context_length:], skip_special_tokens=True),
                model_name_or_path)
            row = {"pred": pred, "answers": json_obj["answers"],
                   "all_classes": json_obj["all_classes"], "length": json_obj["length"],
                   "time": t}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            gc.collect()
            torch.cuda.empty_cache()

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    seed_everything(42)
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    tasks = args.dataset_name_list or LONGBENCH_TASKS
    dataset2prompt = json.load(open(os.path.join(here, "config", "dataset2prompt.json")))
    dataset2maxlen = json.load(open(os.path.join(here, "config", "dataset2maxlen.json")))
    print(f"mode={args.mode} budget={args.budget} tasks={tasks}")

    install_method(args.mode)              # must precede model construction
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    configure_model(model, args)

    run_dir = os.path.join("pred", args.out_name)
    os.makedirs(run_dir, exist_ok=True)
    for task in tasks:
        out_path = os.path.join(run_dir, f"{task}.jsonl")
        if os.path.exists(out_path):
            print(f"{out_path} exists, skipping")
            continue
        data = list(load_longbench_split(args.dataset, task))
        t = time.time()
        get_pred(model, tokenizer, data, args.max_length, dataset2maxlen[task],
                 dataset2prompt[task], task, args.model_name_or_path, out_path, args.mode)
        print(f"== {args.out_name} {task} time: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
