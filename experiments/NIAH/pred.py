import os
from datasets import load_dataset  # noqa: F401 (kept for parity with the other experiments)
import torch
import json
import glob
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
from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_adaptive, replace_llama_adaptive, replace_llama_fixed, replace_mistral_fixed, replace_mistral_slm, replace_llama_slm
from andpro_snapkv.monkeypatch.monkeypatch import replace_mistral_andpro, replace_llama_andpro
from causal_andpro_snapkv.monkeypatch.monkeypatch import replace_mistral_causal, replace_llama_causal
from causal_lookahead_snapkv.monkeypatch.monkeypatch import replace_mistral_lookahead, replace_llama_lookahead
from causal_lookahead_snapkv.monkeypatch.lookahead_runner import generate_with_lookahead

# -----------------------------------------------------------------------------
# Needle-In-A-Haystack (NIAH) prediction driver.
#
# Faithful port of the conventional single-needle test popularized by Greg
# Kamradt (github.com/gkamradt/LLMTest_NeedleInAHaystack): a short "needle"
# fact is hidden at a controlled DEPTH inside a long "haystack" built from the
# Paul Graham essays, the context is grown to a series of target token LENGTHS,
# and the model is asked one retrieval question. Retrieval quality is read off a
# (length x depth) grid / heatmap (see eval.py).
#
# It runs through the *exact same* KV-eviction machinery as LongBench and RULER
# (andpro / causal_forward_only / causal_lookahead), so every config recipe in
# the top-level README works here unchanged. The only difference is data
# construction: the haystack + needle are assembled locally (no HF dataset).
#
# Mapping onto the shared run.sh contract:
#   * the swept "subsets" (--dataset_name_list, i.e. DATASET_NAME_LIST) are the
#     target CONTEXT LENGTHS in tokens (e.g. 4096 8192 16384); one output file
#     pred/<out_name>/<length>.jsonl is written per length.
#   * within each length, the needle is swept over --depth_percents (0..100),
#     one jsonl row per depth.
# -----------------------------------------------------------------------------

# Classic gkamradt single-needle defaults.
DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day."
)
DEFAULT_ANSWER = "eat a sandwich and sit in Dolores Park"
DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"
# Appended to the question, mirroring the original repo's retrieval prompt.
RETRIEVAL_GUIDANCE = "Don't give information outside the document or repeat your findings."

# Bounded backward walk when snapping the insertion point to a sentence
# boundary (mirrors inserters/base.MAX_PERIOD_SNAP in the gkamradt repo).
MAX_PERIOD_SNAP = 100


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True, help="hard cap on the fed prompt length (safety net; target lengths stay below it)")
    parser.add_argument("-d", '--dataset', type=str, default=None, help="unused for NIAH; kept for parity with the shared run.sh contract")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument('--compress_args_path', type=str, default=None, help="unused; kept for parity with LongBench run.sh")
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', "andpro", "critical", 'test', "slm", "causal_forward_only", "causal_lookahead"], help="KV-eviction mode (same set as LongBench/RULER)")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--gqa_func',type=str, default="mean", help="gqa operation:optional max mean")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    parser.add_argument('--budget',default=1024, type=int, help="budget size for kv cache")
    parser.add_argument('--chunk_size',default=4, type=int, help="chunk size for optimization")
    parser.add_argument('--lambda_red', default=0.0, type=float, help="redundancy penalty weight for causal_forward_only's greedy chunk selection (0 disables it)")
    parser.add_argument('--dataset_name_list', default=None, type=str, nargs='+', help="NIAH target CONTEXT LENGTHS in tokens to evaluate, e.g. 4096 8192 16384")

    # --- NIAH-specific knobs (the needle, the question, the haystack, the sweep) ---
    parser.add_argument('--depth_percents', default=None, type=float, nargs='+', help="needle depth percentages to sweep, in [0,100]. Default: 0..100 in steps of 10")
    parser.add_argument('--haystack_dir', default=None, type=str, help="directory of .txt files used to build the haystack (default: ./PaulGrahamEssays)")
    parser.add_argument('--needle', default=DEFAULT_NEEDLE, type=str, help="the fact hidden in the haystack")
    parser.add_argument('--needle_answer', default=DEFAULT_ANSWER, type=str, help="substring that must appear in the model response to count as a retrieval (exact-match scorer)")
    parser.add_argument('--question', default=DEFAULT_QUESTION, type=str, help="retrieval question asked after the context")
    parser.add_argument('--max_gen', default=64, type=int, help="max new tokens generated for the answer")
    parser.add_argument('--final_context_buffer', default=64, type=int, help="tokens reserved below the target length so the needle never spills past it")

    # --- causal_lookahead (Test-Time Lookahead Causal KV Eviction) hyperparameters ---
    parser.add_argument('--num_particles', default=4, type=int, help="[causal_lookahead] M: number of sampled lookahead continuations (Eq. 4)")
    parser.add_argument('--lookahead_len', default=8, type=int, help="[causal_lookahead] T: length of each lookahead continuation (Eq. 4)")
    parser.add_argument('--lookahead_temperature', default=1.0, type=float, help="[causal_lookahead] sampling temperature used only to generate diverse lookahead particles")
    parser.add_argument('--lambda_h', default=0.0, type=float, help="[causal_lookahead] lambda_H: entropy penalty weight in the particle confidence score (Eq. 7)")
    parser.add_argument('--particle_eta', default=1.0, type=float, help="[causal_lookahead] eta: sharpness of the particle-weight softmax (Eq. 8)")
    parser.add_argument('--robust_alpha', default=0.5, type=float, help="[causal_lookahead] alpha: top-beta vs. mean balance in the robust score aggregation (Eq. 17)")
    parser.add_argument('--robust_topk_beta', default=0.5, type=float, help="[causal_lookahead] beta: top-beta fraction of queries used in the robust aggregation (Eq. 16)")
    parser.add_argument('--lambda_lookahead', default=1.0, type=float, help="[causal_lookahead] lambda_L: weight of the particle-weighted lookahead score relative to the window score (Eq. 18)")
    parser.add_argument('--lambda_k', default=0.0, type=float, help="[causal_lookahead] lambda_k: key-similarity diversity penalty in greedy selection (Eq. 21)")
    parser.add_argument('--lambda_v', default=0.0, type=float, help="[causal_lookahead] lambda_v: value-similarity diversity penalty in greedy selection (Eq. 21)")
    parser.add_argument('--budget_tau', default=1.0, type=float, help="[causal_lookahead] tau: how aggressively the head-wise budget favors high-perturbation heads (Eq. 20)")

    parser.add_argument('--budget_ratio',type=float,default=0.2,help="floor_alpha budgets for each head")
    return parser.parse_args(args)


# -----------------------------------------------------------------------------
# Haystack + needle construction (token space, model tokenizer).
#
# NOTE ON TOKENIZATION: the gkamradt repo does all length math with a single
# generic tiktoken encoding (cl100k_base) and lets each provider re-tokenize the
# decoded text. Here we instead use the *model's own* tokenizer for every
# length/insertion computation, exactly as LongBench/RULER pred.py measure
# `context_length` from the model's input_ids. This keeps the reported target
# length faithful to what the KV cache actually holds. The construction logic
# (cycle essays -> trim to length -> snap needle to a sentence boundary ->
# decode -> re-tokenize through the chat template) is otherwise identical.
# -----------------------------------------------------------------------------

def read_haystack_texts(haystack_dir):
    files = sorted(glob.glob(os.path.join(haystack_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt haystack files found under {haystack_dir!r}")
    texts = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def build_context_tokens(essay_token_lists, target_length):
    """Cycle through the pre-tokenized essays until >= target_length, then trim."""
    toks = []
    i = 0
    n = len(essay_token_lists)
    while len(toks) < target_length:
        toks.extend(essay_token_lists[i % n])
        i += 1
    return toks[:target_length]


def snap_to_period(tokenizer, context_tokens, insertion_point, max_snap=MAX_PERIOD_SNAP):
    """Walk backward to the nearest preceding sentence boundary (a token whose
    decoded text contains a period), bounded by max_snap. Robust across
    SentencePiece/BPE tokenizers (we test the decoded piece rather than a fixed
    token id). Returns the original point if none is found in the window."""
    if insertion_point <= 0:
        return 0
    lower_bound = max(0, insertion_point - max_snap)
    while insertion_point > lower_bound:
        piece = tokenizer.decode([context_tokens[insertion_point - 1]])
        if "." in piece:
            return insertion_point
        insertion_point -= 1
    return insertion_point


def insert_needle(tokenizer, context_tokens, needle_tokens, depth_percent):
    """Splice the needle into the context at depth_percent, snapping to a
    sentence boundary for interior depths (mirrors SingleDepthInserter)."""
    pre_len = len(context_tokens)
    if depth_percent >= 100.0:
        insertion_point = pre_len
    elif depth_percent <= 0.0:
        insertion_point = 0
    else:
        insertion_point = int(pre_len * (depth_percent / 100.0))
        insertion_point = snap_to_period(tokenizer, context_tokens, insertion_point)
    new_tokens = context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]
    actual_depth = (insertion_point / pre_len * 100.0) if pre_len > 0 else 0.0
    return new_tokens, insertion_point, actual_depth


def build_niah_input(tokenizer, context_text, question, max_length, device):
    """Assemble the retrieval prompt: the needle'd context followed by the
    question. When the tokenizer exposes a chat template (Mistral/Llama-Instruct)
    we apply it so the format matches the instruct model; otherwise we fall back
    to raw concatenation. Middle-truncation is only a safety net."""
    user_content = f"{context_text}\n\n{question} {RETRIEVAL_GUIDANCE}"
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(user_content, truncation=False, return_tensors="pt").input_ids

    if input_ids.shape[1] > max_length:
        half = int(max_length / 2)
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_pred(model, tokenizer, essay_token_lists, needle_tokens, target_length, depth_percents,
             question, needle_text, needle_answer, max_length, max_gen, final_context_buffer,
             device, out_path, mode=None):
    device = "cuda:0"
    # Build the base haystack once per length (needle is spliced in per depth).
    base_context = build_context_tokens(essay_token_lists, target_length - final_context_buffer - len(needle_tokens))

    rows = []
    with open(f"{out_path}_tmp", "w", encoding="utf-8") as f:
        for depth in tqdm(depth_percents, desc=f"len={target_length}"):
            new_tokens, ins_point, actual_depth = insert_needle(tokenizer, base_context, needle_tokens, depth)
            context_text = tokenizer.decode(new_tokens)
            input_ids, attention_mask = build_niah_input(tokenizer, context_text, question, max_length, device)
            input_length = input_ids.shape[-1]

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

            pred = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            row = {
                "pred": pred,
                # `answer` mirrors RULER (list of gold strings) so string-match tooling is reusable.
                "answer": [needle_answer],
                "needle_answer": needle_answer,
                "needle": needle_text,
                "question": question,
                "task": "niah_single",
                "context_length": target_length,     # target (the swept subset)
                "input_length": input_length,         # actual tokens fed to the model
                "depth_percent": float(depth),        # requested depth
                "actual_depth_percent": actual_depth, # depth after period-snapping
                "length": target_length,
                "time": t,
            }
            rows.append(row)
            json.dump(row, f, ensure_ascii=False)
            f.write('\n')
            f.flush()

            gc.collect()
            torch.cuda.empty_cache()

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
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

    # The swept "subsets" are target context lengths in tokens.
    length_configs = [int(x) for x in args.dataset_name_list] if args.dataset_name_list else [4096, 8192, 16384]
    depth_percents = args.depth_percents if args.depth_percents else [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    haystack_dir = args.haystack_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "PaulGrahamEssays")
    print(f"NIAH lengths={length_configs} depths={depth_percents}")
    print(f"needle={args.needle!r}\nanswer={args.needle_answer!r}\nquestion={args.question!r}\nhaystack_dir={haystack_dir}")

    if not os.path.exists("pred"):
        os.makedirs("pred")

    if args.mode == "ada":
        print("Ada mode")
        replace_mistral_adaptive()
        replace_llama_adaptive()
    elif args.mode == "andpro":
        print("Andpro mode")
        replace_mistral_andpro()
        replace_llama_andpro()
    elif args.mode == "causal_forward_only":
        print("Training-Free Causal AnDPro mode: causal_forward_only")
        replace_mistral_causal()
        replace_llama_causal()
    elif args.mode == "causal_lookahead":
        print("Training-Free Test-Time Lookahead Causal KV Eviction mode: causal_lookahead")
        replace_mistral_lookahead()
        replace_llama_lookahead()
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
    else:
        print("Base mode")

    # config hyperparameters
    def config_compress(model, window_size=32, base_capacity=1024, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, beta = 20, skip=0, gqa_support=False,gqa_func="mean",
                        chunk_size=4, mode="andpro", lambda_red=0.0,
                        num_particles=4, lookahead_len=8, lookahead_temperature=1.0, lambda_h=0.0, particle_eta=1.0,
                        robust_alpha=0.5, robust_topk_beta=0.5, lambda_lookahead=1.0, lambda_k=0.0, lambda_v=0.0, budget_tau=1.0):
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
        model.model.config.lambda_h = lambda_h
        model.model.config.particle_eta = particle_eta
        model.model.config.robust_alpha = robust_alpha
        model.model.config.robust_topk_beta = robust_topk_beta
        model.model.config.lambda_lookahead = lambda_lookahead
        model.model.config.lambda_k = lambda_k
        model.model.config.lambda_v = lambda_v
        model.model.config.budget_tau = budget_tau

        return model

    # NOTE: load model after replace
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    config_compress(model, base_capacity=args.budget, pyram_mode=args.pyram, floor_alpha=args.floor_alpha, gqa_support=args.gqa_support, gqa_func=args.gqa_func,
                    chunk_size=args.chunk_size, mode=args.mode, lambda_red=args.lambda_red,
                    num_particles=args.num_particles, lookahead_len=args.lookahead_len, lookahead_temperature=args.lookahead_temperature,
                    lambda_h=args.lambda_h, particle_eta=args.particle_eta, robust_alpha=args.robust_alpha, robust_topk_beta=args.robust_topk_beta,
                    lambda_lookahead=args.lambda_lookahead, lambda_k=args.lambda_k, lambda_v=args.lambda_v, budget_tau=args.budget_tau)

    # Pre-tokenize the haystack essays and the needle once (model tokenizer, no
    # special tokens). Reused across every length x depth cell.
    essay_texts = read_haystack_texts(haystack_dir)
    essay_token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in essay_texts]
    needle_tokens = tokenizer.encode(args.needle, add_special_tokens=False)
    total_haystack_tokens = sum(len(t) for t in essay_token_lists)
    print(f"loaded {len(essay_texts)} essays, {total_haystack_tokens} haystack tokens, needle={len(needle_tokens)} tokens")

    for length in length_configs:
        if not os.path.exists(f"pred/{args.out_name}"):
            os.makedirs(f"pred/{args.out_name}")
        out_path = f"pred/{args.out_name}/{length}.jsonl"

        if os.path.exists(out_path):
            print(f"{out_path} exists skip")
            continue

        torch.cuda.synchronize()
        t = time.time()
        get_pred(model, tokenizer, essay_token_lists, needle_tokens, length, depth_percents,
                 args.question, args.needle, args.needle_answer, max_length, args.max_gen,
                 args.final_context_buffer, device, out_path, mode=args.mode)
        torch.cuda.synchronize()
        t = time.time() - t
        print(f"== {args.out_name} {length} Time: {t}")
