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
import glob
import shutil
import tempfile
import time
import zipfile
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

# =============================================================================
# Offline LongBench loading
#
# THUDM/LongBench is a loading-script dataset (LongBench.py + a single data.zip
# holding all 35 sub-task jsonl files). datasets>=3 dropped script support, so
# load_dataset("THUDM/LongBench", "<task>") no longer builds anything: it warns
# "couldn't be found on the Hugging Face Hub", falls back to the arrow cache and
# raises "Couldn't find cache for THUDM/LongBench for config '<task>-<hash>'"
# for every sub-task some earlier run never materialised.
#
# Extracting that zip once makes all sub-tasks loadable, online or offline.
# Mirrors figure_making_script/common.py::_longbench_jsonl -- keep the two in
# sync (same extraction dir, so the first of the two to run serves the other).
# =============================================================================
def _hf_cache_roots():
    roots = [os.environ.get("HF_DATASETS_CACHE"), os.environ.get("HF_HOME"),
             os.path.join(os.path.expanduser("~"), ".cache", "huggingface")]
    return [r for r in roots if r]


def _find_longbench_zip():
    hub = os.environ.get("HF_HUB_CACHE")
    hubs = ([hub] if hub else []) + [os.path.join(r, "hub") for r in _hf_cache_roots()]
    for h in hubs:
        hits = sorted(glob.glob(os.path.join(h, "datasets--THUDM--LongBench",
                                             "snapshots", "*", "data.zip")))
        if hits:
            return hits[0]
    return None


def _longbench_jsonl(dataset, dataset_path=None):
    """Path to a local <dataset>.jsonl, or None if we have to go through the Hub."""
    # 1. a local directory handed in via --dataset (dir of jsonl, or its parent)
    if dataset_path and os.path.isdir(dataset_path):
        for cand in (os.path.join(dataset_path, f"{dataset}.jsonl"),
                     os.path.join(dataset_path, "data", f"{dataset}.jsonl")):
            if os.path.exists(cand):
                return cand
    # 2. the hub snapshot's data.zip, extracted once and reused afterwards
    out_dir = os.path.join(_hf_cache_roots()[0], "longbench_jsonl")
    cand = os.path.join(out_dir, f"{dataset}.jsonl")
    if os.path.exists(cand):
        return cand
    zip_path = _find_longbench_zip()
    if zip_path is None:
        return None
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".longbench_", dir=os.path.dirname(out_dir))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        try:
            os.replace(os.path.join(tmp, "data"), out_dir)   # atomic; publishes all 35 files
        except OSError:
            pass                                             # a concurrent job won the race
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cand if os.path.exists(cand) else None


def load_longbench_split(dataset_path, config_name):
    """load_dataset() for one LongBench sub-task, preferring the local jsonl."""
    jsonl = _longbench_jsonl(config_name, dataset_path)
    if jsonl is not None:
        # Same rows and fields as the script build, no Hub round-trip.
        return load_dataset("json", data_files=jsonl, split="train")
    # Hub/arrow-cache path. The data_dir is what produced the hashed offline
    # cache config names (e.g. 'qasper-ae14b6d15afac659'); dropping it misses
    # the cache when running offline.
    return load_dataset(dataset_path, config_name, split="test",
                        data_dir=f"{dataset_path}/data")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("-d", '--dataset', type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
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
    parser.add_argument('--dataset_name_list', default=None, type=str, nargs='+', help="Name(s) of the dataset(s) to use")

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

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        prompt =  [{ "role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
        )
    elif "qwen" in model_name.lower():
        # Qwen2.5-Instruct expects its ChatML template; without it the model
        # continues the document instead of answering. `enable_thinking=False` is
        # inert for Qwen2.5 (its template does not read the variable) and is passed
        # so a hybrid-reasoning Qwen checkpoint cannot silently spend the whole
        # short LongBench budget inside a <think> block.
        prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
        )
    elif "ministral" in model_name.lower():
        # Mistral's instruction format. Deliberately NOT the packaged chat template:
        # that one injects a multi-hundred-token "Le Chat" system prompt (with the
        # current date and tool-use instructions) ahead of every prompt, which would
        # change the long-context input the eviction methods are being compared on.
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "qwen" in model_name.lower():
        # Qwen2.5 emits no reasoning block, but strip one if a future Qwen
        # checkpoint does: a stray </think> otherwise counts as part of the answer
        # string in every LongBench metric.
        if "</think>" in response:
            response = response.split("</think>")[-1]
        response = response.strip()
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name_or_path, out_path, mode=None):
    device = "cuda:0"
    preds = []
    times = []
    with open(f"{out_path}_tmp", "w", encoding="utf-8") as f:
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name_or_path:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name_or_path)
            if "chatglm3" in model_name_or_path:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    input = prompt.to(device)
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]

            torch.cuda.synchronize()
            t = time.time()
            if mode == "causal_lookahead":
                # custom multi-phase generation (prefill -> lookahead particles -> evict -> decode);
                # see causal_lookahead_snapkv.monkeypatch.lookahead_runner.generate_with_lookahead.
                eos_ids = [tokenizer.eos_token_id]
                if dataset == "samsum":
                    eos_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
                answer_ids = generate_with_lookahead(
                    model, tokenizer, input.input_ids, input.attention_mask, max_gen, eos_ids,
                )
                output = torch.cat([input.input_ids[0], answer_ids], dim=0)
            elif dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
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
            pred = post_process(pred, model_name_or_path)
            preds.append(pred)


            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "time": t}, f, ensure_ascii=False )
            f.write('\n')
            f.flush()


            gc.collect()
            torch.cuda.empty_cache()


    with open(out_path, "w", encoding="utf-8") as f:
        for json_obj, pred, t in zip(data, preds, times):
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "time": t}, f, ensure_ascii=False )
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
    # define your model
    max_length = args.max_length
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    datasets = [
                "qasper", "narrativeqa", "multifieldqa_en", # single doc
                "hotpotqa", "2wikimqa", "musique",          # multi doc
                "trec", "triviaqa", "samsum",               # few-shot
                "gov_report", "qmsum", "multi_news",        # sum
                "passage_count", "passage_retrieval_en",    # Synthetic
                "lcc", "repobench-p",                       # code
                ]

    if args.dataset_name_list:
        datasets = args.dataset_name_list
        
    print(datasets)
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

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

    for dataset in datasets:
        if args.e:
            data = load_longbench_split(args.dataset, f"{dataset}_e")
            if not os.path.exists(f"pred_e/{args.out_name}"):
                os.makedirs(f"pred_e/{args.out_name}")
            out_path = f"pred_e/{args.out_name}/{dataset}.jsonl"
        else:
            data = load_longbench_split(args.dataset, f"{dataset}")
            if not os.path.exists(f"pred/{args.out_name}"):
                os.makedirs(f"pred/{args.out_name}")
            out_path = f"pred/{args.out_name}/{dataset}.jsonl"

        if os.path.exists(out_path):
            print(f"{out_path} exists skip")
            continue

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        # TODO: hard code single process, which use all gpus
        torch.cuda.synchronize()
        t = time.time()
        get_pred(model, tokenizer, data_all, max_length, max_gen, prompt_format, dataset, device, model_name_or_path, out_path, mode=args.mode)
        torch.cuda.synchronize()
        t = time.time() - t
        print(f"== {args.out_name} {dataset} Time: {t}")
