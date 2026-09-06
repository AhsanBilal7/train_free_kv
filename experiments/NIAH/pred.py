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
# Needle-In-A-Haystack (NIAH) prediction driver.
#
# Faithful port of the test popularized by Greg Kamradt
# (github.com/gkamradt/needle-in-a-haystack). Short "needle" facts are hidden at
# controlled DEPTHs inside a long "haystack" built from the Paul Graham essays,
# the context is grown to a series of target token LENGTHS, and the model is
# asked one retrieval question. Retrieval quality is read off a (length x depth)
# grid / heatmap (see eval.py).
#
# Two tasks, matching needlehaystack/tasks/ in the reference repo:
#   * "single" (SingleNeedleTask + SingleDepthInserter) -- one fact at one depth.
#   * "multi"  (MultiNeedleTask  + EvenSpreadInserter)  -- N facts spread evenly
#     from depth_percent to 100%; scored as the FRACTION of expected fragments
#     that appear in the response (ExactMatchScorer).
#
# It runs through the *exact same* KV-eviction machinery as LongBench and RULER
# (andpro / causal_forward_only / causal_lookahead), so every config recipe in
# the top-level README works here unchanged. The only difference is data
# construction: the haystack + needles are assembled locally (no HF dataset).
#
# Mapping onto the shared run.sh contract:
#   * the swept "subsets" (--dataset_name_list, i.e. DATASET_NAME_LIST) are the
#     target CONTEXT LENGTHS in tokens (e.g. 4096 8192 16384); one output file
#     pred/<out_name>/<length>.jsonl is written per length.
#   * within each length, the needles are swept over --depth_percents (0..100),
#     one jsonl row per depth.
#
# DELIBERATE DEVIATIONS FROM THE REFERENCE (all recorded in each output row):
#   1. Tokenization. The reference does every length computation with one
#      generic tiktoken encoding (cl100k_base) and lets the provider
#      re-tokenize. We use the MODEL'S OWN tokenizer throughout, exactly as
#      LongBench/RULER pred.py measure context_length from the model's
#      input_ids, so the reported length reflects what the KV cache holds.
#   2. Length accounting. The reference trims the haystack to exactly `length`
#      and then inserts the needles ON TOP (final = length + needle tokens, and
#      the question is not counted at all). We budget the other way round so the
#      TOTAL fed to the model lands at ~`length`: haystack = length - buffer -
#      sum(needle tokens). What a KV-eviction method sees is then exactly the
#      swept length, which is the quantity this repo cares about.
# -----------------------------------------------------------------------------

# Classic gkamradt single-needle defaults (needlehaystack/tasks/single_needle.py).
DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day."
)
DEFAULT_ANSWER = "eat a sandwich and sit in Dolores Park"
DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"

# Multi-needle defaults (needlehaystack/tasks/multi_needle.py): the pizza trio.
# `DEFAULT_FRAGMENTS` are what the scorer looks for, one per needle.
DEFAULT_NEEDLES_MULTI = [
    "Figs are one of the secret ingredients needed to build the perfect pizza.",
    "Prosciutto is one of the secret ingredients needed to build the perfect pizza.",
    "Goat cheese is one of the secret ingredients needed to build the perfect pizza.",
]
DEFAULT_FRAGMENTS_MULTI = ["figs", "prosciutto", "goat cheese"]
DEFAULT_QUESTION_MULTI = "What are the secret ingredients needed to build the perfect pizza?"

# Bounded backward walk when snapping the insertion point to a sentence
# boundary (mirrors inserters/base.MAX_PERIOD_SNAP in the gkamradt repo).
MAX_PERIOD_SNAP = 100

# NOTE ON THE PROMPT: the reference runner (core/runner.py) builds exactly
#     user_prompt = f"{context}\n\n{question}"
# and sends system="". Older (v1) versions of the repo appended
#     "Don't give information outside the document or repeat your findings"
# to the question. That string is NOT in the reference implementation this
# harness targets, and on Mistral-7B-Instruct-v0.2 it is actively harmful: the
# model frequently *echoes it as the answer* ("The best thing to do in San
# Francisco is to not give information outside the document..."), which scores
# as a retrieval failure that has nothing to do with the KV cache. It also eats
# ~13 of the 32 observation-window tokens that every SnapKV-family method scores
# the prefix against. Default is therefore "" (reference behaviour); pass
# --retrieval_guidance to opt back into the v1 string.
DEFAULT_RETRIEVAL_GUIDANCE = ""


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

    # --- NIAH-specific knobs (the needles, the question, the haystack, the sweep) ---
    parser.add_argument('--task', default='single', choices=['single', 'multi'],
                        help="'single' = one needle at one depth (SingleDepthInserter); "
                             "'multi' = N needles spread evenly from depth to 100%% (EvenSpreadInserter)")
    parser.add_argument('--depth_percents', default=None, type=float, nargs='+', help="needle depth percentages to sweep, in [0,100]. Default: 0..100 in steps of 10")
    parser.add_argument('--haystack_dir', default=None, type=str, help="directory of .txt files used to build the haystack (default: ./PaulGrahamEssays)")
    parser.add_argument('--needle', default=None, type=str, help="[single] the fact hidden in the haystack")
    parser.add_argument('--needle_answer', default=None, type=str, help="[single] substring that must appear in the model response to count as a retrieval (exact-match scorer)")
    parser.add_argument('--needles', default=None, type=str, nargs='+', help="[multi] the facts hidden in the haystack, one per --needle_fragments entry")
    parser.add_argument('--needle_fragments', default=None, type=str, nargs='+', help="[multi] substrings scored against the response; the cell score is the FRACTION found (reference ExactMatchScorer)")
    parser.add_argument('--question', default=None, type=str, help="retrieval question asked after the context")
    parser.add_argument('--retrieval_guidance', default=DEFAULT_RETRIEVAL_GUIDANCE, type=str,
                        help="extra sentence appended after the question. Default '' matches the reference runner; "
                             "pass the v1 string \"Don't give information outside the document or repeat your findings.\" to restore old behaviour")
    parser.add_argument('--max_gen', default=64, type=int, help="max new tokens generated for the answer")
    parser.add_argument('--final_context_buffer', default=64, type=int, help="tokens reserved below the target length for the question + chat template, so the prompt never overshoots it")
    parser.add_argument('--on_overflow', default='error', choices=['error', 'truncate'],
                        help="what to do when the assembled prompt exceeds --max_length. 'error' (default) aborts: "
                             "middle-truncation DELETES the needle for interior depths and silently invalidates the cell. "
                             "'truncate' restores the old (unsafe) behaviour")

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


def build_period_token_ids(tokenizer, max_scan=None):
    """Token ids whose decoded piece contains a period.

    The reference walks back to a token in `set(tiktoken.encode("."))`. With a
    SentencePiece/BPE vocab a sentence end is spelled by many different ids
    ('.', '.\\n', '.)', '."', ...), so we build the equivalent set once per run
    instead of calling tokenizer.decode() up to MAX_PERIOD_SNAP times per cell.
    Falls back to a decode-based probe if the vocab cannot be enumerated."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    ids = set()
    for piece, idx in vocab.items():
        if "." in piece:
            ids.add(idx)
    return ids or None


def snap_to_period(context_tokens, insertion_point, period_ids, max_snap=MAX_PERIOD_SNAP):
    """Walk backward to the nearest preceding sentence boundary, bounded by
    max_snap. `insertion_point` means "insert BEFORE the token at this index",
    so we test the token immediately preceding it -- identical semantics to
    inserters/base.snap_to_period. Returns the original point if no period is
    found inside the window (the reference's v1 code looped unbounded)."""
    if insertion_point <= 0:
        return 0
    lower_bound = max(0, insertion_point - max_snap)
    while insertion_point > lower_bound:
        if context_tokens[insertion_point - 1] in period_ids:
            return insertion_point
        insertion_point -= 1
    return insertion_point


def insert_needle_single(context_tokens, needle_token_lists, depth_percent, period_ids):
    """SingleDepthInserter: splice ONE needle at depth_percent, snapping to a
    sentence boundary for interior depths. Returns
    (new_tokens, [insertion_index], [actual_depth_percent])."""
    if len(needle_token_lists) != 1:
        raise ValueError(f"task='single' takes exactly 1 needle; got {len(needle_token_lists)}")
    needle_tokens = needle_token_lists[0]
    pre_len = len(context_tokens)
    if depth_percent >= 100.0:
        insertion_point = pre_len
    elif depth_percent <= 0.0:
        insertion_point = 0
    else:
        insertion_point = int(pre_len * (depth_percent / 100.0))
        insertion_point = snap_to_period(context_tokens, insertion_point, period_ids)
    new_tokens = context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]
    # Depth is measured against the PRE-insertion length so it stays comparable
    # across needles and runs (inserters/single_depth.py).
    actual_depth = (insertion_point / pre_len * 100.0) if pre_len > 0 else 0.0
    return new_tokens, [insertion_point], [actual_depth]


def insert_needles_even_spread(context_tokens, needle_token_lists, depth_percent, period_ids):
    """EvenSpreadInserter: place needle 0 at depth_percent and distribute the
    rest evenly through the remaining context, stopping short of 100% (the
    interval spans n slots, not n-1, so the last needle is not pinned to the very
    end). Line-for-line port of inserters/even_spread.py, including its two fixes
    over the v1 multi-needle code:

      * every target index and reported depth is computed against the
        PRE-insertion length, so earlier needles do not inflate later depths;
      * the depth>=100 branch still records a placement.

    Returns (new_tokens, insertion_indices, actual_depth_percents)."""
    if not needle_token_lists:
        raise ValueError("task='multi' needs at least one needle")
    pre_len = len(context_tokens)
    n = len(needle_token_lists)

    interval = (100.0 - depth_percent) / n
    target_depths = [depth_percent + i * interval for i in range(n)]

    planned = []
    for toks, depth in zip(needle_token_lists, target_depths):
        if depth >= 100.0:
            raw_index = pre_len
        else:
            raw_index = int(pre_len * (depth / 100.0))
            raw_index = snap_to_period(context_tokens, raw_index, period_ids)
        actual_depth = (raw_index / pre_len * 100.0) if pre_len > 0 else 0.0
        planned.append((toks, raw_index, actual_depth))

    # target_depths is monotonically non-decreasing, so `planned` is already in
    # ascending pre-insertion index order; splice in that order and carry the
    # accumulated shift so later indices stay valid.
    new_tokens = list(context_tokens)
    indices, depths = [], []
    accumulated_shift = 0
    for toks, raw_index, actual_depth in planned:
        final_index = raw_index + accumulated_shift
        new_tokens = new_tokens[:final_index] + toks + new_tokens[final_index:]
        indices.append(final_index)
        depths.append(actual_depth)
        accumulated_shift += len(toks)
    return new_tokens, indices, depths


def build_niah_input(tokenizer, context_text, question, guidance, max_length, device, on_overflow):
    """Assemble the retrieval prompt.

    The reference runner builds `f"{context}\\n\\n{question}"` with an empty
    system message; `guidance` (default "") appends the optional v1 sentence.
    When the tokenizer exposes a chat template (Mistral/Llama-Instruct) we apply
    it so the format matches the instruct model; otherwise we fall back to raw
    concatenation.

    Overflow is NOT a benign safety net here. Middle-truncation drops the centre
    of the document, which is precisely where the needle sits for every interior
    depth -- the cell then measures nothing and still gets recorded under its
    nominal target length. So we abort by default."""
    tail = f"{question} {guidance}".strip() if guidance else question
    user_content = f"{context_text}\n\n{tail}"
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            return_tensors="pt",
            # See build_ruler_input in experiments/RULER/pred.py: the
            # non-thinking switch, ignored by templates that do not use it.
            enable_thinking=False,
        )
    else:
        input_ids = tokenizer(user_content, truncation=False, return_tensors="pt").input_ids

    if input_ids.shape[1] > max_length:
        if on_overflow == "error":
            raise ValueError(
                f"assembled prompt is {input_ids.shape[1]} tokens but --max_length is {max_length}. "
                f"Middle-truncating would delete the needle for interior depths and silently "
                f"invalidate this cell. Raise --max_length (NIAH_MAX_LENGTH) above the largest "
                f"target length, or pass --on_overflow truncate to accept the old behaviour."
            )
        half = int(max_length / 2)
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_pred(model, tokenizer, essay_token_lists, needle_token_lists, target_length, depth_percents,
             question, guidance, needle_texts, expected_fragments, task, period_ids,
             max_length, max_gen, final_context_buffer, device, out_path, mode=None, seed=42,
             on_overflow="error"):
    device = "cuda:0"
    inserter = insert_needle_single if task == "single" else insert_needles_even_spread
    inserter_name = "single_depth" if task == "single" else "even_spread"

    # Build the base haystack once per length (needles are spliced in per depth).
    # Budget so the TOTAL prompt lands at ~target_length: the buffer covers the
    # question + chat-template wrapper that get_pred appends afterwards.
    needle_budget = sum(len(t) for t in needle_token_lists)
    haystack_len = target_length - final_context_buffer - needle_budget
    if haystack_len <= 0:
        raise ValueError(
            f"target length {target_length} is too small for {len(needle_token_lists)} needle(s) "
            f"({needle_budget} tokens) + buffer ({final_context_buffer})"
        )
    base_context = build_context_tokens(essay_token_lists, haystack_len)

    tmp_path = f"{out_path}_tmp"

    # --- Resume: reuse cells already completed by a prior interrupted run ---
    # (same contract as RULER's pred.py). Count only fully-flushed, valid-JSON
    # lines; a truncated final line means the process died mid-write, so we stop
    # there and regenerate that cell onward.
    #
    # A _tmp file left by a run with a DIFFERENT prompt/needle/depth grid must
    # not be resumed into this one -- that would silently mix cells generated
    # under two different conditions in a single output file. Each candidate row
    # is therefore checked against the current configuration, and everything
    # from the first mismatch onward is regenerated.
    def _row_matches(row, cell_idx):
        if cell_idx >= len(depth_percents):
            return False
        return (
            row.get("task") == f"niah_{task}"
            and row.get("question") == question
            and row.get("retrieval_guidance", "") == guidance
            and row.get("needles") == list(needle_texts)
            and row.get("expected_fragments") == list(expected_fragments)
            and row.get("context_length") == target_length
            and float(row.get("depth_percent", -1)) == float(depth_percents[cell_idx])
        )

    rows = []
    if os.path.exists(tmp_path):
        stale = False
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break  # partial trailing line -> resume from here
                if not _row_matches(row, len(rows)):
                    stale = True
                    break
                rows.append(row)
        if stale:
            print(f"{tmp_path}: rows are stale (different prompt/needle/depth grid) — "
                  f"discarding {'all' if not rows else f'from cell {len(rows)}'} and regenerating")
        if rows:
            print(f"{tmp_path}: resuming from {len(rows)}/{len(depth_percents)} completed cells")
        # Truncate whatever we refused to reuse so the append below stays consistent.
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')

    # Append so previously-completed cells are preserved (do NOT truncate).
    with open(tmp_path, "a", encoding="utf-8") as f:
        for cell_idx, depth in enumerate(tqdm(depth_percents, desc=f"len={target_length}")):
            if cell_idx < len(rows):
                continue
            # Re-seed per cell. causal_lookahead samples its M lookahead
            # particles with torch.multinomial, so without this a cell's result
            # depends on how many cells ran before it -- two methods (or two
            # resumed runs) would then see different RNG streams on the same
            # (length, depth) cell. Deterministic modes are unaffected.
            seed_everything(seed + 1000 * cell_idx)

            new_tokens, ins_points, actual_depths = inserter(
                base_context, needle_token_lists, depth, period_ids
            )
            context_text = tokenizer.decode(new_tokens)
            input_ids, attention_mask = build_niah_input(
                tokenizer, context_text, question, guidance, max_length, device, on_overflow
            )
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
                # `expected_fragments` is what eval.py scores: the cell score is
                # the FRACTION of these found in the response (ExactMatchScorer).
                # For task='single' it is a 1-element list, so the score is the
                # classic 1.0/0.0 exact match.
                "expected_fragments": list(expected_fragments),
                # `answer` mirrors RULER (list of gold strings) so string-match tooling is reusable.
                "answer": list(expected_fragments),
                "needles": list(needle_texts),
                "question": question,
                "retrieval_guidance": guidance,
                "task": f"niah_{task}",
                "inserter": inserter_name,
                "context_length": target_length,       # target (the swept subset)
                "input_length": input_length,           # actual tokens fed to the model
                "haystack_token_count": haystack_len,   # needle-free haystack size
                "depth_percent": float(depth),          # requested depth
                # depth(s) after period-snapping, measured against the
                # pre-insertion haystack. Lists, one entry per needle.
                "actual_depth_percents": actual_depths,
                "insertion_token_indices": ins_points,
                # scalar mirror of the first needle, kept so existing tooling
                # (and eval.py's grid) keeps working unchanged.
                "actual_depth_percent": actual_depths[0],
                "length": target_length,
                "seed": seed + 1000 * cell_idx,
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

    # --- Resolve the task (mirrors needlehaystack/tasks/{single,multi}_needle.py) ---
    if args.task == "single":
        needle_texts = [args.needle if args.needle is not None else DEFAULT_NEEDLE]
        expected_fragments = [args.needle_answer if args.needle_answer is not None else DEFAULT_ANSWER]
        question = args.question if args.question is not None else DEFAULT_QUESTION
    else:
        needle_texts = args.needles if args.needles else list(DEFAULT_NEEDLES_MULTI)
        expected_fragments = args.needle_fragments if args.needle_fragments else list(DEFAULT_FRAGMENTS_MULTI)
        question = args.question if args.question is not None else DEFAULT_QUESTION_MULTI
        if not needle_texts:
            raise ValueError("task='multi' needs at least one --needles entry")
        if not expected_fragments:
            raise ValueError("task='multi' needs at least one --needle_fragments entry")
        if args.needles and args.needle_fragments and len(args.needles) != len(args.needle_fragments):
            # Not fatal in the reference either, but a mismatch almost always
            # means a quoting mistake in run.sh, and it silently changes the
            # denominator of the fractional score.
            print(f"[warn] {len(args.needles)} needles but {len(args.needle_fragments)} fragments — "
                  f"cell scores are fractions over the FRAGMENT count")

    print(f"NIAH task={args.task} lengths={length_configs} depths={depth_percents}")
    print(f"needles={needle_texts}\nfragments={expected_fragments}\nquestion={question!r}\n"
          f"guidance={args.retrieval_guidance!r}\nhaystack_dir={haystack_dir}")

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

    # Pre-tokenize the haystack essays and the needles once (model tokenizer, no
    # special tokens). Reused across every length x depth cell.
    essay_texts = read_haystack_texts(haystack_dir)
    essay_token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in essay_texts]
    needle_token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in needle_texts]
    period_ids = build_period_token_ids(tokenizer)
    if period_ids is None:
        raise RuntimeError("could not enumerate the tokenizer vocab to build the period-token set")
    total_haystack_tokens = sum(len(t) for t in essay_token_lists)
    print(f"loaded {len(essay_texts)} essays, {total_haystack_tokens} haystack tokens, "
          f"needles={[len(t) for t in needle_token_lists]} tokens, "
          f"{len(period_ids)} period-bearing token ids")

    # --- Guard the two ways a swept length silently stops measuring retrieval ---
    # (a) prompt > --max_length  -> middle-truncation deletes the needle;
    # (b) prompt > the model's positional range -> RoPE extrapolates and the
    #     whole grid row degrades for reasons unrelated to KV eviction.
    max_pos = getattr(model.config, "max_position_embeddings", None)
    over_cap = [L for L in length_configs if L > max_length]
    if over_cap:
        msg = (f"target length(s) {over_cap} exceed --max_length {max_length}; the prompt would be "
               f"middle-truncated and the needle deleted for interior depths")
        if args.on_overflow == "error":
            raise ValueError(msg + ". Raise NIAH_MAX_LENGTH, or pass --on_overflow truncate.")
        print(f"[warn] {msg} (--on_overflow truncate: results for those lengths are NOT valid NIAH cells)")
    if max_pos:
        over_pos = [L for L in length_configs if L > max_pos]
        if over_pos:
            print(f"[warn] target length(s) {over_pos} exceed the model's max_position_embeddings "
                  f"({max_pos}); those rows measure positional extrapolation, not KV eviction, "
                  f"and are not comparable to the in-range rows.")

    if total_haystack_tokens < max(length_configs):
        print(f"[warn] haystack is only {total_haystack_tokens} tokens but the largest target is "
              f"{max(length_configs)}; the essays will be cycled (duplicated) to fill the context, "
              f"as the reference FilesHaystack also does.")

    for length in length_configs:
        if not os.path.exists(f"pred/{args.out_name}"):
            os.makedirs(f"pred/{args.out_name}")
        out_path = f"pred/{args.out_name}/{length}.jsonl"

        if os.path.exists(out_path):
            print(f"{out_path} exists skip")
            continue

        torch.cuda.synchronize()
        t = time.time()
        get_pred(model, tokenizer, essay_token_lists, needle_token_lists, length, depth_percents,
                 question, args.retrieval_guidance, needle_texts, expected_fragments,
                 args.task, period_ids, max_length, args.max_gen,
                 args.final_context_buffer, device, out_path, mode=args.mode,
                 on_overflow=args.on_overflow)
        torch.cuda.synchronize()
        t = time.time() - t
        print(f"== {args.out_name} {length} Time: {t}")
