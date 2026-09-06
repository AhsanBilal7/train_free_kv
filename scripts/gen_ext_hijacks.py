"""Generate <pkg>/monkeypatch/*_ext_hijack.py from the package's Llama hijack.

The new backbones (Qwen3, Ministral3) reuse the Llama attention forward verbatim
except for the q/k/v projection block, so the file is derived from the Llama one
by textual substitution rather than retyped -- any drift would be a silent
behaviour change on a new backbone only.
"""
import re, sys, os

REPO = "/home/u.ab353540/train_free_kv"

# (package, source hijack file, attention forward name, init symbol, model-forward symbol,
#  prepare-inputs symbol, output file, generated forward name)
JOBS = [
    ("andpro_snapkv", "andpro_llama_hijack.py", "andpro_llama_flash_attn2_forward",
     "init_andpro_snapkv", "andpro_LlamaModel_forward", "prepare_inputs_for_generation_llama",
     "andpro_ext_hijack.py", "andpro_ext_flash_attn2_forward"),
    ("adaptive_snapkv", "adaptive_llama_hijack.py", "adaptive_llama_flash_attn2_forward",
     "init_adaptive_snapkv", "adaptive_LlamaModel_forward", "prepare_inputs_for_generation_llama",
     "adaptive_ext_hijack.py", "adaptive_ext_flash_attn2_forward"),
    ("causal_andpro_snapkv", "causal_llama_hijack.py", "causal_llama_flash_attn2_forward",
     "init_andpro_snapkv", "causal_LlamaModel_forward", "prepare_inputs_for_generation_llama",
     "causal_ext_hijack.py", "causal_ext_flash_attn2_forward"),
    ("causal_lookahead_snapkv", "causal_llama_hijack.py", "causal_lookahead_llama_flash_attn2_forward",
     "init_lookahead_snapkv", "causal_lookahead_LlamaModel_forward",
     "prepare_inputs_for_generation_lookahead_llama",
     "causal_lookahead_ext_hijack.py", "causal_lookahead_ext_flash_attn2_forward"),
]

# The block every Llama hijack uses to build q/k/v, which the new backbones replace
# with the model's own `project_qkv` (QK-norm for Qwen3) plus an optional post-RoPE
# query scale (the llama-4 length scaling for Ministral3).
PROJ_RE = re.compile(
    r"[ \t]*query_states = self\.q_proj\(hidden_states\)\n"
    r"[ \t]*key_states = self\.k_proj\(hidden_states\)\n"
    r"[ \t]*value_states = self\.v_proj\(hidden_states\)\n"
    r"(?:[ \t]*#[^\n]*\n)*"
    r"[ \t]*\n?"
    r"(?:[ \t]*#[^\n]*\n)*"
    r"[ \t]*query_states = query_states\.view\([^\n]*\n"
    r"[ \t]*key_states = key_states\.view\([^\n]*\n"
    r"[ \t]*value_states = value_states\.view\([^\n]*\n"
)

PROJ_NEW = (
    "    # The one place these backbones differ from Llama: Qwen3 RMS-normalizes q and k\n"
    "    # per head before RoPE, and both models size their heads from config.head_dim.\n"
    "    # `project_qkv` is defined on the model's own attention class (models_ext), so\n"
    "    # this single forward serves every backbone registered there.\n"
    "    query_states, key_states, value_states = self.project_qkv(hidden_states)\n"
)

SCALE_NEW = (
    "    # Ministral3 scales the query by 1 + beta*log(1 + floor(pos/16384)) after RoPE;\n"
    "    # backbones without that scaling do not define the hook and are unaffected.\n"
    "    if hasattr(self, \"scale_query\"):\n"
    "        query_states = self.scale_query(query_states, position_ids)\n"
)

HEADER = '''"""{fwd} -- {pkg}'s attention hijack for the models_ext backbones.

GENERATED from {pkg}/monkeypatch/{src} by
scripts/gen_ext_hijacks.py; do not edit by hand. Re-run the generator after any
change to the Llama hijack so the two stay in lockstep.

Qwen3 and Ministral3 are expressed in the Llama idiom (see models_ext/), so the
model body, the causal mask, the cache and `prepare_inputs_for_generation` are
all inherited from Llama and reused as-is by the replace_* functions in
{pkg}/monkeypatch/monkeypatch.py. Only the attention forward has to be
re-emitted, because the vendored attention classes define their own `forward`
and would otherwise shadow whatever is patched onto LlamaFlashAttention2.

The body below is the Llama one with a single edit: the three q/k/v projections
and their `.view(...).transpose(1, 2)` reshapes are replaced by the backbone's
`project_qkv`, and an optional post-RoPE `scale_query` hook is applied.
"""
'''


def extract(src_text, name):
    """The full source of `def <name>(...)` up to the next top-level def."""
    start = src_text.index(f"def {name}(")
    rest = src_text[start:]
    m = re.search(r"\n(?=def |\Z)", rest[1:])
    end = len(rest) if m is None else m.start() + 1
    return rest[:end].rstrip() + "\n"


for pkg, src, fwd_name, init_sym, model_fwd, prep_fn, out_name, new_fwd in JOBS:
    src_path = os.path.join(REPO, pkg, "monkeypatch", src)
    text = open(src_path).read()
    body = extract(text, fwd_name)

    body, n = PROJ_RE.subn(PROJ_NEW, body)
    assert n == 1, f"{pkg}: projection block matched {n} times"

    # Insert the query scale right after RoPE is applied.
    rope_line = "    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)\n"
    assert body.count(rope_line) == 1, f"{pkg}: rope line not found exactly once"
    body = body.replace(rope_line, rope_line + SCALE_NEW)

    # `self.hidden_size` in the decode reshape assumes num_heads*head_dim == hidden_size,
    # which holds for these checkpoints but not for every Qwen3 size; -1 is equivalent
    # where it holds and correct where it does not.
    body = body.replace("reshape(bsz, q_len, self.hidden_size)", "reshape(bsz, q_len, -1)")

    body = body.replace(f"def {fwd_name}(", f"def {new_fwd}(")

    imports = [
        "from typing import List, Optional, Tuple, Union",
        "",
        "import torch",
        "from transformers.cache_utils import Cache, DynamicCache, StaticCache",
        "from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv",
        "from transformers.modeling_flash_attention_utils import _flash_attention_forward",
        "from transformers.utils import logging, is_flash_attn_2_available",
        "",
        f"from {pkg}.monkeypatch.snapkv_utils import {init_sym}",
        "",
        "logger = logging.get_logger(__name__)",
        "",
        "if is_flash_attn_2_available():",
        "    from flash_attn import flash_attn_func, flash_attn_varlen_func",
        "",
        "",
    ]
    out = HEADER.format(fwd=new_fwd, pkg=pkg, src=src) + "\n" + "\n".join(imports) + "\n" + body
    out_path = os.path.join(REPO, pkg, "monkeypatch", out_name)
    open(out_path, "w").write(out)
    print("wrote", out_path, len(out.splitlines()), "lines")
