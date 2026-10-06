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
from causal_lookahead_snapkv.monkeypatch.hybrid_baseline_hijack import replace_hybrid_baseline
from causal_lookahead_snapkv.monkeypatch.monkeypatch import replace_mistral_lookahead, replace_llama_lookahead, replace_qwen2_lookahead, replace_ministral3_lookahead, replace_hybrid_lookahead
from causal_lookahead_snapkv.monkeypatch.lookahead_runner import generate_with_lookahead

# Ministral3 postdates the transformers pin (see models_ext/); registering it with
# the Auto* factories is what lets its checkpoint resolve at all. It is a no-op for
# every backbone transformers already knows -- Qwen2 among them -- so it runs
# unconditionally and must run before load_model_and_tokenizer.
from models_ext import register_extra_architectures


def _compress_targets(model):
    """Every config object the eviction code may read compression settings from.

    config_compress() historically wrote to `model.model.config` only. That is the
    same object the attention modules see on a flat ForCausalLM -> model -> layers
    stack, but NOT on a multimodal one: Gemma4UnifiedTextAttention holds a
    Gemma4UnifiedTextConfig while `model.model.config` is the top-level multimodal
    config, so every setting landed somewhere the layers never look
    ("'Gemma4UnifiedTextConfig' object has no attribute 'base_capacity'").

    Returns the whole chain -- top-level, model-level, and any nested text_config --
    so a write cannot miss the object that is actually read. Writing to all of them
    is safe: they are the same values, and on a flat stack the chain collapses to
    one object.
    """
    seen, out = {}, []
    for cfg in (getattr(model, "config", None),
                getattr(getattr(model, "model", None), "config", None)):
        while cfg is not None and id(cfg) not in seen:
            seen[id(cfg)] = True
            out.append(cfg)
            cfg = getattr(cfg, "text_config", None)
    return out


def _set_compress_attr(model, name, value):
    for cfg in _compress_targets(model):
        setattr(cfg, name, value)



# The 4.44-only backbone patches (mistral/llama/qwen2/ministral3) monkeypatch
# *FlashAttention2, which transformers 4.48 deleted. pred.py calls every replace_*
# unconditionally, so on the transformers 5.x venv used by the hybrid-attention
# backbones those calls raise AttributeError even though the run does not want them.
#
# Gate on ONE explicit capability probe rather than try/except around each call: a
# blanket except would also swallow a genuine AttributeError on 4.44, where these
# patches must run and a failure means something is really broken.
def _legacy_flash_attn_stack():
    import transformers.models.mistral.modeling_mistral as _m
    return hasattr(_m, "MistralFlashAttention2")


_LEGACY = _legacy_flash_attn_stack()


def _legacy(*fns):
    """Run the 4.44-only backbone patches, or skip them all on a 5.x stack."""
    if not _LEGACY:
        return
    for fn in fns:
        fn()


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
    # 'full' and 'test' are the same thing -- neither installs a hijack, so the model
    # decodes from the complete KV cache. 'full' is the name to use: it is what the
    # Full Cache row of every results table is, and out_name is built from this string,
    # so a run stored as test_<model>_... reads like a throwaway rather than the
    # uncompressed reference. 'test' is kept only so older commands still parse.
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', "andpro", "critical", 'full', 'test', "slm", "causal_forward_only", "causal_lookahead"], help="Ada mode, fix mode, andpro mode, criticalkv mode, full (uncompressed reference; 'test' is a legacy alias), causal_forward_only (Training-Free Causal AnDPro), causal_lookahead (Training-Free Test-Time Lookahead Causal KV Eviction), or normal")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--gqa_func',type=str, default="mean", help="gqa operation:optional max mean")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    parser.add_argument('--budget',default=1024, type=int, help="budget size for kv cache")
    parser.add_argument('--chunk_size',default=4, type=int, help="chunk size for optimization")
    parser.add_argument('--lambda_red', default=0.0, type=float, help="redundancy penalty weight for causal_forward_only's greedy chunk selection (0 disables it)")
    parser.add_argument('--dataset_name_list', default=None, type=str, nargs='+', help="RULER sequence-length config(s) to evaluate, e.g. 4096 8192 16384 65536 131072")
    parser.add_argument('--shard', default=os.environ.get("RULER_SHARD") or None, type=str,
                        help="k/K: decode only rows i >= shard_base with (i - shard_base) %% K == k (see merge_shards)")
    parser.add_argument('--shard_base', default=int(os.environ.get("RULER_SHARD_BASE", "0")), type=int,
                        help="rows < shard_base were already decoded by the unsharded run (<len>.jsonl_tmp)")
    parser.add_argument('--local_dataset', default=os.environ.get("RULER_LOCAL_DIR"), type=str,
                        help="Directory of locally generated RULER configs (long_context/make_data.sh), "
                             "read for every length the Hub dataset does not ship (anything but 4096/8192/16384)")

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
        # A RULER needle can sit anywhere, so middle truncation can delete it and the
        # row then scores as a model failure. This must never fire on a real run --
        # raise MAX_LENGTH instead (long_context/check_lengths.py sizes it).
        print(f"WARNING: prompt of {input_ids.shape[1]} tokens truncated to max_length={max_length}")
        half = int(max_length / 2)
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

def get_pred(model, tokenizer, data, max_length, device, model_name_or_path, out_path, mode=None, tmp_path=None):
    device = "cuda:0"
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
        # while the file grows. A kill mid-write is routine here (gpu_debug's 2 h
        # wall clock lands mid-row), so the torn tail must be removed, not skipped.
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

# Sequence-length configs the Hub dataset (simonjegou/ruler) actually ships.
HUB_LENGTHS = {"4096", "8192", "16384"}


def load_ruler_config(hub_dataset, local_dir, length):
    """One RULER length config as a list of rows.

    4096/8192/16384 come from the Hub exactly as before. Longer configs do not
    exist there; they are generated with NVIDIA RULER by long_context/make_data.sh
    into `local_dir`, in the same schema and by the same generator revision, seed
    and tokenizer (long_context/validate.py checks this against the Hub rows).
    Read as parquet directly: no dataset script or network involved.
    """
    if length in HUB_LENGTHS:
        return load_dataset(hub_dataset, length, split='test')
    path = os.path.join(local_dir or "", length, "test-00000-of-00001.parquet")
    if not local_dir or not os.path.exists(path):
        raise FileNotFoundError(
            f"RULER config {length} is not on the Hub ({sorted(HUB_LENGTHS, key=int)}) and no local "
            f"copy was found at {path!r}. Generate it with "
            f"experiments/RULER/long_context/make_data.sh and set RULER_LOCAL_DIR.")
    return load_dataset("parquet", data_files={"test": path}, split="test")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def _attn_impl(model_path=None):
    """Which attention kernel to ask transformers to build the model with.

    "flash_attention_2" when flash-attn is genuinely importable AND is not the
    import-only stub, AND the model's head_dim is inside what the kernel supports
    through transformers' autograd-wrapped entry point.

    head_dim 256 is the trap: flash-attn 2.7.4 accepts it for a bare forward, but
    transformers routes through an autograd custom op whose backward is limited to
    192, and the guard fires at forward time --
        RuntimeError: FlashAttention forward only supports head dimension at most 256
    Gemma 4 and Qwen3.5 both have head_dim 256, and Gemma 4's *sliding* layers hit
    it first because they fall through to the stock forward.

    Falling back to "sdpa" is not a degradation of the method: the prefill and
    lookahead phases dispatch through
    ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, ...), so the
    kernel choice is transparent there, and the post-eviction ragged decode calls
    flash_attn_varlen_func directly (see _varlen_attend) regardless of this setting.
    """
    try:
        import flash_attn
        if getattr(flash_attn, "__lorekv_stub__", False):
            return "sdpa"
    except Exception:
        return "sdpa"
    if model_path:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path)
            tcfg = getattr(cfg, "text_config", None) or cfg
            hd = tcfg.to_dict().get("head_dim")
            if isinstance(hd, (list, tuple)):
                hd = max(hd)
            if hd and int(hd) > 192:
                return "sdpa"
        except Exception:
            pass
    return "flash_attention_2"


def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              device_map="cuda:0",
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                             torch_dtype=torch.bfloat16,
                                             # TODO: hard code
                                             device_map="cuda:0",
                                             attn_implementation=_attn_impl(path),
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
        _legacy(replace_mistral_adaptive, replace_llama_adaptive, replace_qwen2_adaptive)
        replace_hybrid_baseline('adaptive')
        _legacy(replace_ministral3_adaptive)
        replace_hybrid_baseline('adaptive')
    elif args.mode == "andpro":
        print("Andpro mode")
        _legacy(replace_mistral_andpro, replace_llama_andpro, replace_qwen2_andpro)
        replace_hybrid_baseline('andpro')
        _legacy(replace_ministral3_andpro)
        replace_hybrid_baseline('andpro')
    elif args.mode == "causal_forward_only":
        print("Training-Free Causal AnDPro mode: causal_forward_only")
        _legacy(replace_mistral_causal, replace_llama_causal, replace_qwen2_causal, replace_ministral3_causal)
    elif args.mode == "causal_lookahead":
        print("Training-Free Test-Time Lookahead Causal KV Eviction mode: causal_lookahead")
        _legacy(replace_mistral_lookahead, replace_llama_lookahead, replace_qwen2_lookahead)
        replace_hybrid_lookahead()
        _legacy(replace_ministral3_lookahead)
        replace_hybrid_lookahead()
    elif args.mode == "fix":
        print("Fix mode")
        _legacy(replace_mistral_fixed, replace_llama_fixed)
    elif args.mode == "slm":
        print("Slm mode")
        _legacy(replace_mistral_slm, replace_llama_slm)
    elif args.mode == "critical":
        print("Criticalkv mode")
        _legacy(replace_mistral_adaptive, replace_llama_adaptive, replace_qwen2_adaptive)
        replace_hybrid_baseline('adaptive')
        _legacy(replace_ministral3_adaptive)
        replace_hybrid_baseline('adaptive')
    else:
        print("Base mode")

    # config hyperparameters

    def config_compress(model, window_size=32, base_capacity=1024, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, beta = 20, skip=0, gqa_support=False,gqa_func="mean",
                        chunk_size=4, mode="andpro", lambda_red=0.0,
                        num_particles=4, lookahead_len=8, lookahead_temperature=1.0, particle_beta=1.0, particle_gamma=0.0, lambda_d=0.0, normalize_typicality=False,
                        robust_alpha=0.5, robust_topk_beta=0.5, lambda_lookahead=1.0, lambda_k=0.0, lambda_v=0.0, budget_tau=1.0,
                        score_objective="projected"):
        _set_compress_attr(model, "window_size", window_size)
        _set_compress_attr(model, "base_capacity", base_capacity)
        _set_compress_attr(model, "kernel_size", kernel_size)

        _set_compress_attr(model, "pooling", pooling)
        _set_compress_attr(model, "floor_alpha", floor_alpha)
        _set_compress_attr(model, "skip", skip)
        _set_compress_attr(model, "normalize", None)

        _set_compress_attr(model, "pyram_mode", pyram_mode)
        _set_compress_attr(model, "pyram_beta", beta)

        _set_compress_attr(model, "gqa_support", gqa_support)
        _set_compress_attr(model, "gqa_func", gqa_func)

        _set_compress_attr(model, "chunk_size", chunk_size)
        _set_compress_attr(model, "mode", mode)
        _set_compress_attr(model, "lambda_red", lambda_red)

        # causal_lookahead (Test-Time Lookahead Causal KV Eviction) hyperparameters.
        _set_compress_attr(model, "lookahead_phase", "collect")
        _set_compress_attr(model, "lookahead_particle_idx", 0)
        _set_compress_attr(model, "num_particles", num_particles)
        _set_compress_attr(model, "lookahead_len", lookahead_len)
        _set_compress_attr(model, "lookahead_temperature", lookahead_temperature)
        _set_compress_attr(model, "particle_beta", particle_beta)
        _set_compress_attr(model, "particle_gamma", particle_gamma)
        _set_compress_attr(model, "lambda_d", lambda_d)
        _set_compress_attr(model, "normalize_typicality", normalize_typicality)
        _set_compress_attr(model, "robust_alpha", robust_alpha)
        _set_compress_attr(model, "robust_topk_beta", robust_topk_beta)
        _set_compress_attr(model, "lambda_lookahead", lambda_lookahead)
        _set_compress_attr(model, "lambda_k", lambda_k)
        _set_compress_attr(model, "lambda_v", lambda_v)
        _set_compress_attr(model, "budget_tau", budget_tau)
        _set_compress_attr(model, "score_objective", score_objective)

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
        data = load_ruler_config(args.dataset, args.local_dataset, str(length))
        if not os.path.exists(f"pred/{args.out_name}"):
            os.makedirs(f"pred/{args.out_name}")
        out_path = f"pred/{args.out_name}/{length}.jsonl"

        if os.path.exists(out_path):
            print(f"{out_path} exists skip")
            continue

        # attach the config length to each row so it lands in the output jsonl
        data_all = [{**data_sample, "length": int(length)} for data_sample in data]
        if args.shard:
            k, K = (int(x) for x in args.shard.split("/"))
            run_dir = f"pred/{args.out_name}"
            part = f"{run_dir}/{length}.shard{k}of{K}.jsonl_part"
            if not os.path.exists(part):
                rows = [{**data_all[i], "idx": i} for i in range(args.shard_base, len(data_all))
                        if (i - args.shard_base) % K == k]
                print(f"shard {k}/{K} of {run_dir}/{length}: {len(rows)} rows from base {args.shard_base}")
                get_pred(model, tokenizer, rows, max_length, device, model_name_or_path, part,
                         mode=args.mode, tmp_path=part + "_tmp")
            merge_shards(run_dir, length, K, args.shard_base, len(data_all))
            continue
        torch.cuda.synchronize()
        t = time.time()
        get_pred(model, tokenizer, data_all, max_length, device, model_name_or_path, out_path, mode=args.mode)
        torch.cuda.synchronize()
        t = time.time() - t
        print(f"== {args.out_name} {length} Time: {t}")
