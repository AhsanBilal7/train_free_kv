import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_adaptive, replace_llama_adaptive, replace_llama_fixed, replace_mistral_fixed, replace_mistral_slm, replace_llama_slm, replace_qwen2_adaptive, replace_ministral3_adaptive
from andpro_snapkv.monkeypatch.monkeypatch import replace_mistral_andpro, replace_llama_andpro, replace_qwen2_andpro, replace_ministral3_andpro
from causal_andpro_snapkv.monkeypatch.monkeypatch import replace_mistral_causal, replace_llama_causal, replace_qwen2_causal, replace_ministral3_causal
from causal_lookahead_snapkv.monkeypatch.monkeypatch import replace_mistral_lookahead, replace_llama_lookahead, replace_qwen2_lookahead, replace_ministral3_lookahead
from causal_lookahead_snapkv.monkeypatch.lookahead_runner import generate_with_lookahead

# Ministral3 postdates the transformers pin (see models_ext/); registering it with
# the Auto* factories is what lets its checkpoint resolve at all. It is a no-op for
# every backbone transformers already knows -- Qwen2 among them -- so it runs
# unconditionally and must run before load_model_and_tokenizer.
from models_ext import register_extra_architectures

register_extra_architectures()

# -----------------------------------------------------------------------------
# RULER prediction driver.
#
# Mirrors experiments/LongBench/pred.py so RULER runs through the *exact* same KV
# eviction machinery (andpro / causal_forward_only / causal_lookahead), but the
# input format is RULER's (simonjegou/ruler on the Hub): every row already ships
# its own prompt pieces -- `context`, `question`, `answer_prefix`, gold `answer`
# (a list), `task` and `max_new_tokens` -- so there is no config/dataset2prompt
# lookup. The RULER "subsets" swept over `--dataset_name_list` are the sequence-
# length configs ("4096", "8192", "16384"), each of which bundles all 13 tasks.
# -----------------------------------------------------------------------------

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument("-d", '--dataset', type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args (unused; kept for parity with LongBench run.sh)")
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', "andpro", "critical", 'test', "slm", "causal_forward_only", "causal_lookahead"], help="Ada mode, fix mode, andpro mode, criticalkv mode, causal_forward_only (Training-Free Causal AnDPro), causal_lookahead (Training-Free Test-Time Lookahead Causal KV Eviction), or normal")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--gqa_func',type=str, default="mean", help="gqa operation:optional max mean")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    parser.add_argument('--budget',default=1024, type=int, help="budget size for kv cache")
    parser.add_argument('--chunk_size',default=4, type=int, help="chunk size for optimization")
    parser.add_argument('--lambda_red', default=0.0, type=float, help="redundancy penalty weight for causal_forward_only's greedy chunk selection (0 disables it)")
    parser.add_argument('--dataset_name_list', default=None, type=str, nargs='+', help="RULER sequence-length config(s) to evaluate, e.g. 4096 8192 16384")

    # --- causal_lookahead (Test-Time Lookahead Causal KV Eviction) hyperparameters ---
    parser.add_argument('--num_particles', default=4, type=int, help="[causal_lookahead] M: number of sampled lookahead continuations (Eq. 4)")
    parser.add_argument('--lookahead_len', default=8, type=int, help="[causal_lookahead] T: length of each lookahead continuation (Eq. 4)")
    parser.add_argument('--lookahead_temperature', default=1.0, type=float, help="[causal_lookahead] sampling temperature used only to generate diverse lookahead particles")
    parser.add_argument('--beta', default=1.0, type=float, help="[causal_lookahead] beta: TYPICALITY coefficient in Eq. 3 -- weights the centered per-token score c_u = log q(y_u|h_u) + H(q(.|h_u)), whose conditional expectation is zero, so this channel prices branch atypicality with the entropy removed")
    parser.add_argument('--gamma', default=0.0, type=float, help="[causal_lookahead] gamma: ENTROPY coefficient in Eq. 3 -- an explicit preference for low-uncertainty histories. gamma = 1 + lambda_H_old reproduces a published run")
    parser.add_argument('--normalize_typicality', default=0, type=int, choices=[0, 1], help="[causal_lookahead] use the varentropy-normalized typicality ctilde = cbar / sqrt(Vbar/T) instead of cbar (Prop. 3). Ablated switch, off by default")
    parser.add_argument('--lambda_d', default=0.0, type=float, help="[causal_lookahead] lambda_D: trajectory-disagreement coefficient in Eq. 3 -- how atypical particle m's future-query trajectory is w.r.t. the other particles. Identically 0 for M <= 2")
    parser.add_argument('--robust_alpha', default=0.5, type=float, help="[causal_lookahead] alpha: top-beta vs. mean balance in the robust score aggregation (Eq. 17)")
    parser.add_argument('--robust_topk_beta', default=0.5, type=float, help="[causal_lookahead] beta: top-beta fraction of queries used in the robust aggregation (Eq. 16)")
    parser.add_argument('--lambda_lookahead', default=1.0, type=float, help="[causal_lookahead] lambda_L: weight of the particle-weighted lookahead score relative to the window score (Eq. 18)")
    parser.add_argument('--lambda_k', default=0.0, type=float, help="[causal_lookahead] lambda_k: key-similarity diversity penalty in greedy selection (Eq. 21)")
    parser.add_argument('--lambda_v', default=0.0, type=float, help="[causal_lookahead] lambda_v: value-similarity diversity penalty in greedy selection (Eq. 21)")
    parser.add_argument('--budget_tau', default=1.0, type=float, help="[causal_lookahead] tau: how aggressively the head-wise budget favors high-perturbation heads (Eq. 20)")
    parser.add_argument('--score_objective', default="projected", type=str, choices=["projected", "unprojected", "attention"], help="[causal_lookahead] App. C (iii) perturbation-objective ablation: 'projected' = ||W_O Delta o||, the Eq. 14 core score; 'unprojected' = ||Delta o||, the same leave-one-out effect without the output projection; 'attention' = accumulated attention mass a_{q,i}. All three use the identical window + lookahead queries, reliability weights, allocation and selection")

    parser.add_argument('--budget_ratio',type=float,default=0.2,help="floor_alpha budgets for each head")
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
        half = int(max_length / 2)
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

def get_pred(model, tokenizer, data, max_length, device, model_name_or_path, out_path, mode=None):
    device = "cuda:0"
    tmp_path = f"{out_path}_tmp"

    # --- Resume: reuse rows already completed by a prior interrupted run ---
    # The _tmp file streams one JSON object per finished row (flushed each step).
    # Count only fully-flushed (valid-JSON) lines; a truncated final line means the
    # process died mid-write, so we stop there and regenerate that row onward.
    preds = []
    times = []
    done = 0
    if os.path.exists(tmp_path):
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    break  # partial trailing line -> resume from here
                preds.append(obj["pred"])
                times.append(obj["time"])
                done += 1
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
                # see causal_lookahead_snapkv.monkeypatch.lookahead_runner.generate_with_lookahead.
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

            json.dump({"pred": pred, "answer": json_obj["answer"], "task": json_obj["task"],
                       "length": json_obj["length"], "context_length": context_length, "time": t}, f, ensure_ascii=False)
            f.write('\n')
            f.flush()

            gc.collect()
            torch.cuda.empty_cache()

    with open(out_path, "w", encoding="utf-8") as f:
        for json_obj, pred, t in zip(data, preds, times):
            json.dump({"pred": pred, "answer": json_obj["answer"], "task": json_obj["task"],
                       "length": json_obj["length"], "time": t}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              device_map="cuda:0",
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                             torch_dtype=torch.bfloat16,
                                             # TODO: hard code
                                             device_map="cuda:0",
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = args.model_name_or_path
    model_name = args.model_name_or_path.split("/")[-1]
    max_length = args.max_length

    # RULER "subsets" are sequence-length configs on the Hub, each bundling all 13 tasks.
    length_configs = args.dataset_name_list if args.dataset_name_list else ["4096", "8192", "16384"]
    print(length_configs)

    if not os.path.exists("pred"):
        os.makedirs("pred")

    if args.mode == "ada":
        print("Ada mode")
        replace_mistral_adaptive()
        replace_llama_adaptive()
        replace_qwen2_adaptive()
        replace_ministral3_adaptive()
    elif args.mode == "andpro":
        print("Andpro mode")
        replace_mistral_andpro()
        replace_llama_andpro()
        replace_qwen2_andpro()
        replace_ministral3_andpro()
    elif args.mode == "causal_forward_only":
        print("Training-Free Causal AnDPro mode: causal_forward_only")
        replace_mistral_causal()
        replace_llama_causal()
        replace_qwen2_causal()
        replace_ministral3_causal()
    elif args.mode == "causal_lookahead":
        print("Training-Free Test-Time Lookahead Causal KV Eviction mode: causal_lookahead")
        replace_mistral_lookahead()
        replace_llama_lookahead()
        replace_qwen2_lookahead()
        replace_ministral3_lookahead()
    elif args.mode == "fix":
        print("Fix mode")
        replace_mistral_fixed()
        replace_llama_fixed()
    elif args.mode == "slm":
        print("Slm mode")
        replace_mistral_slm()
        replace_llama_slm()
    elif args.mode == "critical":
        print("Criticalkv mode")
        replace_mistral_adaptive()
        replace_llama_adaptive()
        replace_qwen2_adaptive()
        replace_ministral3_adaptive()
    else:
        print("Base mode")

    # config hyperparameters
    def config_compress(model, window_size=32, base_capacity=1024, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, beta = 20, skip=0, gqa_support=False,gqa_func="mean",
                        chunk_size=4, mode="andpro", lambda_red=0.0,
                        num_particles=4, lookahead_len=8, lookahead_temperature=1.0, particle_beta=1.0, particle_gamma=0.0, lambda_d=0.0, normalize_typicality=False,
                        robust_alpha=0.5, robust_topk_beta=0.5, lambda_lookahead=1.0, lambda_k=0.0, lambda_v=0.0, budget_tau=1.0,
                        score_objective="projected"):
        model.model.config.window_size = window_size
        model.model.config.base_capacity = base_capacity
        model.model.config.kernel_size = kernel_size

        model.model.config.pooling = pooling
        model.model.config.floor_alpha = floor_alpha
        model.model.config.skip = skip
        model.model.config.normalize = None

        model.model.config.pyram_mode = pyram_mode
        model.model.config.pyram_beta = beta

        model.model.config.gqa_support = gqa_support
        model.model.config.gqa_func = gqa_func

        model.model.config.chunk_size = chunk_size
        model.model.config.mode = mode
        model.model.config.lambda_red = lambda_red

        # causal_lookahead (Test-Time Lookahead Causal KV Eviction) hyperparameters.
        model.model.config.lookahead_phase = "collect"
        model.model.config.lookahead_particle_idx = 0
        model.model.config.num_particles = num_particles
        model.model.config.lookahead_len = lookahead_len
        model.model.config.lookahead_temperature = lookahead_temperature
        model.model.config.particle_beta = particle_beta
        model.model.config.particle_gamma = particle_gamma
        model.model.config.lambda_d = lambda_d
        model.model.config.normalize_typicality = normalize_typicality
        model.model.config.robust_alpha = robust_alpha
        model.model.config.robust_topk_beta = robust_topk_beta
        model.model.config.lambda_lookahead = lambda_lookahead
        model.model.config.lambda_k = lambda_k
        model.model.config.lambda_v = lambda_v
        model.model.config.budget_tau = budget_tau
        model.model.config.score_objective = score_objective

        return model

    # NOTE: load model after replace
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    config_compress(model, base_capacity=args.budget, pyram_mode=args.pyram, floor_alpha=args.floor_alpha, gqa_support=args.gqa_support, gqa_func=args.gqa_func,
                    chunk_size=args.chunk_size, mode=args.mode, lambda_red=args.lambda_red,
                    num_particles=args.num_particles, lookahead_len=args.lookahead_len, lookahead_temperature=args.lookahead_temperature,
                    particle_beta=args.beta, particle_gamma=args.gamma, lambda_d=args.lambda_d,
                    normalize_typicality=bool(args.normalize_typicality),
                    robust_alpha=args.robust_alpha, robust_topk_beta=args.robust_topk_beta,
                    lambda_lookahead=args.lambda_lookahead, lambda_k=args.lambda_k, lambda_v=args.lambda_v, budget_tau=args.budget_tau,
                    score_objective=args.score_objective)

    for length in length_configs:
        data = load_dataset(args.dataset, str(length), split='test')
        if not os.path.exists(f"pred/{args.out_name}"):
            os.makedirs(f"pred/{args.out_name}")
        out_path = f"pred/{args.out_name}/{length}.jsonl"

        if os.path.exists(out_path):
            print(f"{out_path} exists skip")
            continue

        # attach the config length to each row so it lands in the output jsonl
        data_all = [{**data_sample, "length": int(length)} for data_sample in data]
        torch.cuda.synchronize()
        t = time.time()
        get_pred(model, tokenizer, data_all, max_length, device, model_name_or_path, out_path, mode=args.mode)
        torch.cuda.synchronize()
        t = time.time() - t
        print(f"== {args.out_name} {length} Time: {t}")
