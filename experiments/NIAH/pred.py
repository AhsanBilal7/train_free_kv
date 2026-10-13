r"""Needle-in-a-Haystack (NIAH) prediction driver (paper Sec. 4.2, Figure 8).

The driver implements the test of Greg Kamradt
(https://github.com/gkamradt/LLMTest_NeedleInAHaystack, MIT). It puts short facts
("needles") at specified depths in a haystack made from the Paul Graham essays.
It makes the context for each target length and asks one retrieval question.
``eval.py`` scores the results as a grid of length and depth.

Tasks (as in ``needlehaystack/tasks/`` of the reference):

* ``single`` (SingleNeedleTask, SingleDepthInserter): one fact at one depth.
* ``multi`` (MultiNeedleTask, EvenSpreadInserter): N facts between the depth and
  100%. The score is the fraction of expected fragments in the answer.

Each target length (``--dataset_name_list``) gives one file
``pred/<out_name>/<length>.jsonl`` with one row for each depth. All modes of
:mod:`lorekv.harness` are supported.

Differences from the reference (each output row records them):

1. Tokenization. The reference measures lengths with tiktoken ``cl100k_base``.
   This driver uses the tokenizer of the model, so the length is the number of
   tokens in the KV cache.
2. Length. The reference adds the needles to a haystack of ``length`` tokens.
   This driver uses haystack = length - buffer - needle tokens, so the full
   prompt has ``length`` tokens.
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lorekv.harness import (add_method_args, configure_model, install_method,  # noqa: E402
                            load_model_and_tokenizer, seed_everything)
from lorekv.monkeypatch.lookahead_runner import generate_with_lookahead  # noqa: E402


# Single-needle defaults of the reference (needlehaystack/tasks/single_needle.py).
DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day."
)
DEFAULT_ANSWER = "eat a sandwich and sit in Dolores Park"
DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"

# Multi-needle defaults of the reference (needlehaystack/tasks/multi_needle.py).
# The scorer looks for one fragment for each needle.
DEFAULT_NEEDLES_MULTI = [
    "Figs are one of the secret ingredients needed to build the perfect pizza.",
    "Prosciutto is one of the secret ingredients needed to build the perfect pizza.",
    "Goat cheese is one of the secret ingredients needed to build the perfect pizza.",
]
DEFAULT_FRAGMENTS_MULTI = ["figs", "prosciutto", "goat cheese"]
DEFAULT_QUESTION_MULTI = "What are the secret ingredients needed to build the perfect pizza?"

# Maximum number of tokens to move back to a sentence end
# (inserters/base.MAX_PERIOD_SNAP of the reference).
MAX_PERIOD_SNAP = 100

# The reference runner (core/runner.py) uses user_prompt = f"{context}\n\n{question}"
# and an empty system message. Older versions added "Don't give information outside
# the document or repeat your findings" to the question. Mistral-7B often repeats that
# sentence as its answer, and the sentence uses about 13 of the 32 observation-window
# tokens. The default is thus "" (as in the reference). Use --retrieval_guidance to
# add the sentence.
DEFAULT_RETRIEVAL_GUIDANCE = ""


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True, help="hard cap on the fed prompt length (safety net; target lengths stay below it)")
    parser.add_argument("-d", '--dataset', type=str, default=None, help="unused for NIAH; kept for parity with the shared run.sh contract")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument('--dataset_name_list', default=None, type=str, nargs='+', help="NIAH target CONTEXT LENGTHS in tokens to evaluate, e.g. 4096 8192 16384")

    # NIAH settings: needles, question, haystack and depths.
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

    add_method_args(parser)
    return parser.parse_args(args)


# -----------------------------------------------------------------------------
# Haystack + needle construction (token space, model tokenizer).
#
# The reference computes all lengths with tiktoken (cl100k_base). This driver uses the
# tokenizer of the model, so the target length is the number of tokens in the KV cache.
# The other steps are the same: repeat the essays, cut to the length, move the needle to
# a sentence end, decode, and tokenize again with the chat template.
# -----------------------------------------------------------------------------

def read_haystack_texts(haystack_dir):
    """Read the text of each ``.txt`` file in ``haystack_dir``, in sorted order."""
    files = sorted(glob.glob(os.path.join(haystack_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt haystack files found under {haystack_dir!r}")
    texts = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def build_context_tokens(essay_token_lists, target_length):
    """Repeat the tokenized essays until there are ``target_length`` tokens, then cut."""
    toks = []
    i = 0
    n = len(essay_token_lists)
    while len(toks) < target_length:
        toks.extend(essay_token_lists[i % n])
        i += 1
    return toks[:target_length]


def build_period_token_ids(tokenizer, max_scan=None):
    """Return the IDs of the tokens whose text contains a period.

    The reference uses ``set(tiktoken.encode("."))``. A SentencePiece or BPE
    vocabulary has many tokens that end a sentence ('.', '.\\n', '.)', ...), so the
    function finds them one time for each run. If the vocabulary is not available,
    it uses ``tokenizer.decode`` on each token.
    """
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
    """Move the insertion point back to the nearest sentence end, by at most ``max_snap`` tokens.

    ``insertion_point`` means "insert before this index", so the function tests the
    token before it, as in ``inserters/base.snap_to_period``. If no period is found,
    the function returns the original point.
    """
    if insertion_point <= 0:
        return 0
    lower_bound = max(0, insertion_point - max_snap)
    while insertion_point > lower_bound:
        if context_tokens[insertion_point - 1] in period_ids:
            return insertion_point
        insertion_point -= 1
    return insertion_point


def insert_needle_single(context_tokens, needle_token_lists, depth_percent, period_ids):
    """Insert one needle at ``depth_percent`` (SingleDepthInserter).

    For a depth between 0% and 100%, the insertion point moves to a sentence end.

    Returns:
        A tuple ``(new_tokens, [insertion_index], [actual_depth_percent])``.
    """
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
    # Measure the depth against the length before insertion (inserters/single_depth.py).
    actual_depth = (insertion_point / pre_len * 100.0) if pre_len > 0 else 0.0
    return new_tokens, [insertion_point], [actual_depth]


def insert_needles_even_spread(context_tokens, needle_token_lists, depth_percent, period_ids):
    """Insert needle 0 at ``depth_percent`` and the other needles at equal intervals after it.

    This follows ``inserters/even_spread.py`` of the reference. The interval uses n
    slots, so the last needle is not at the end of the context. The target indices
    and depths use the length before insertion.

    Returns:
        A tuple ``(new_tokens, insertion_indices, actual_depth_percents)``.
    """
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

    # `planned` is in ascending index order. Insert in this order and add the shift
    # of the earlier needles to each later index.
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
    """Make the token IDs of the retrieval prompt.

    The reference uses ``f"{context}\\n\\n{question}"`` and an empty system message.
    ``guidance`` (default "") adds an optional sentence. If the tokenizer has a chat
    template, the function applies it.

    If the prompt is longer than ``max_length``, the function raises an error by
    default. Middle truncation removes the center of the context, which contains
    the needle at most depths.
    """
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
    """Decode each depth of one target length and write the rows to ``out_path``."""
    device = "cuda:0"
    inserter = insert_needle_single if task == "single" else insert_needles_even_spread
    inserter_name = "single_depth" if task == "single" else "even_spread"

    # Make the haystack one time for each length. The buffer leaves space for the
    # question and the chat template, so the full prompt has about target_length tokens.
    needle_budget = sum(len(t) for t in needle_token_lists)
    haystack_len = target_length - final_context_buffer - needle_budget
    if haystack_len <= 0:
        raise ValueError(
            f"target length {target_length} is too small for {len(needle_token_lists)} needle(s) "
            f"({needle_budget} tokens) + buffer ({final_context_buffer})"
        )
    base_context = build_context_tokens(essay_token_lists, haystack_len)

    tmp_path = f"{out_path}_tmp"

    # Continue an interrupted run, as in RULER pred.py. Use only complete, valid rows.
    # Each row must also have the current prompt, needles and depth. Decode again from
    # the first row that does not agree.
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
        # Remove the rows that are not used, so that the appended rows follow the valid ones.
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')

    # Append, so that the finished cells stay in the file.
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


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    device = "cuda:0"
    model_name_or_path = args.model_name_or_path
    max_length = args.max_length

    # The target context lengths in tokens.
    length_configs = [int(x) for x in args.dataset_name_list] if args.dataset_name_list else [4096, 8192, 16384]
    depth_percents = args.depth_percents if args.depth_percents else [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    haystack_dir = args.haystack_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "PaulGrahamEssays")

    # Resolve the task (as in needlehaystack/tasks/{single,multi}_needle.py).
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

    install_method(args.mode)              # must precede model construction
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    configure_model(model, args)

    # Tokenize the essays and the needles one time, without special tokens.
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

    # Check two conditions that make a length invalid:
    # (a) prompt > --max_length: middle truncation removes the needle.
    # (b) prompt > the position range of the model: RoPE extrapolates.
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
