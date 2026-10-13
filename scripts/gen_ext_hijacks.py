"""🛠️ Generate the extra-backbone attention hijacks from each package's Llama hijack.

Two backbones need one, for different reasons, and both are derived by textual
substitution from the package's own Llama hijack rather than retyped -- any drift
between them would be a silent behaviour change on the new backbone only.

  <pkg>/monkeypatch/<name>_ext_hijack.py     Ministral3 (models_ext)
      The vendored classes subclass Llama, so the model forward, the causal mask,
      the cache and prepare_inputs_for_generation are all inherited and reused
      as-is. Only the attention forward is re-emitted, because the vendored
      attention class defines its own `forward` and would otherwise shadow
      whatever is patched onto LlamaFlashAttention2. The one edit is to route the
      q/k/v projections through the backbone's `project_qkv` hook and to apply its
      optional post-RoPE `scale_query` hook.

  <pkg>/monkeypatch/<name>_qwen2_hijack.py   Qwen2 (native in transformers 4.44)
      Qwen2 needs no vendoring, but its 4.44 implementation predates the rotary
      refactor: the attention forward takes no `position_embeddings`, the rotary
      module is table-based (`rotary_emb(x, seq_len=...)`), and
      `apply_rotary_pos_emb` indexes that table with `position_ids`. The model
      forward is otherwise the Llama one -- same `_update_causal_mask` signature --
      minus the `position_embeddings` it computes and threads through the layers,
      so both functions are emitted here.

Re-run after any change to a Llama hijack so the copies stay in lockstep:
    python scripts/gen_ext_hijacks.py
"""

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (package, llama hijack file, attention forward, model forward, init symbol,
#  module that defines the init symbol and DynamicCacheSplitHeadFlatten)
JOBS = [
    ("andpro_snapkv", "andpro_llama_hijack.py",
     "andpro_llama_flash_attn2_forward", "andpro_LlamaModel_forward", "init_andpro_snapkv",
     "snapkv_utils"),
    ("adaptive_snapkv", "adaptive_llama_hijack.py",
     "adaptive_llama_flash_attn2_forward", "adaptive_LlamaModel_forward", "init_adaptive_snapkv",
     "snapkv_utils"),
    ("lorekv", "causal_llama_hijack.py",
     "causal_lookahead_llama_flash_attn2_forward", "causal_lookahead_LlamaModel_forward",
     "init_lookahead_snapkv", "scoring"),
]

# The q/k/v block every Llama hijack uses, replaced for the models_ext backbones.
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
    "    # The one place this backbone differs from Llama: it sizes its heads from\n"
    "    # config.head_dim. `project_qkv` is defined on the model's own attention class\n"
    "    # (models_ext), so this single forward serves every backbone registered there.\n"
    "    query_states, key_states, value_states = self.project_qkv(hidden_states)\n"
)

SCALE_NEW = (
    "    # Ministral3 scales the query by 1 + beta*log(1 + floor(pos/16384)) after RoPE;\n"
    "    # backbones without that scaling do not define the hook and are unaffected.\n"
    "    if hasattr(self, \"scale_query\"):\n"
    "        query_states = self.scale_query(query_states, position_ids)\n"
)

# The Llama rotary block, in both the terse and the deprecation-warning variants.
ROPE_RE = re.compile(
    r"[ \t]*if position_embeddings is None:\n"
    r"(?:.*?\n)*?"
    r"[ \t]*query_states, key_states = apply_rotary_pos_emb\(query_states, key_states, cos, sin\)\n"
)

ROPE_QWEN2 = (
    "    # Qwen2's rotary module is table-based: it returns cos/sin for [0, seq_len) and\n"
    "    # apply_rotary_pos_emb then indexes that table with position_ids. seq_len must\n"
    "    # therefore come from the positions being served, NOT from the cache: in the\n"
    "    # post-eviction phase the flattened per-head cache reports a length of 1 while\n"
    "    # position_ids keeps counting, and the table would be short by thousands of rows.\n"
    "    # The model forward computes it once per call so this is not a per-layer sync.\n"
    "    rope_seq_len = getattr(self.config, \"_rope_seq_len\", None)\n"
    "    if rope_seq_len is None:\n"
    "        rope_seq_len = int(position_ids.max()) + 1\n"
    "    cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)\n"
    "    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)\n"
)

SW_RE = re.compile(r'([ \t]*)sliding_window=getattr\(self, "sliding_window", None\),\n')
SW_QWEN2 = (
    '\\1sliding_window=(\n'
    '\\1    self.config.sliding_window\n'
    '\\1    if (self.config.use_sliding_window\n'
    '\\1        and getattr(self.config, "sliding_window", None) is not None\n'
    '\\1        and self.layer_idx >= self.config.max_window_layers)\n'
    '\\1    else None\n'
    '\\1),\n'
)

ROPE_STASH = (
    "\n"
    "    # Qwen2's rotary is table-based and apply_rotary_pos_emb indexes the table with\n"
    "    # position_ids, so the table has to reach the largest position actually served.\n"
    "    # Computed once here rather than in each attention layer: reading it off a device\n"
    "    # tensor is a synchronization, and 48 layers x every decode step adds up.\n"
    "    self.config._rope_seq_len = int(position_ids.max()) + 1\n"
)

HEADER = '''"""{names} -- {pkg}'s hijack{plural} for {backbone}.

GENERATED from {pkg}/monkeypatch/{src} by scripts/gen_ext_hijacks.py;
do not edit by hand. Re-run the generator after any change to the Llama hijack
so the two stay in lockstep. See that script's docstring for what is edited and
why.
"""
'''


def extract(src_text, name):
    """The full source of `def <name>(...)`, up to the next top-level def."""
    start = src_text.index(f"def {name}(")
    rest = src_text[start:]
    m = re.search(r"\n(?=def |\Z)", rest[1:])
    end = len(rest) if m is None else m.start() + 1
    return rest[:end].rstrip() + "\n"


def drop_position_embeddings_arg(body):
    """Remove the `position_embeddings` parameter -- Qwen2's 4.44 forward has none."""
    out, n = re.subn(r"[ \t]*position_embeddings: Optional\[Tuple\[[^\]]*\]\][^\n]*\n", "", body)
    assert n == 1, f"position_embeddings parameter matched {n} times"
    return out


def emit(path, header, imports, bodies):
    text = header + "\n" + "\n".join(imports) + "\n" + "\n\n".join(bodies)
    open(path, "w").write(text)
    print("wrote", path, len(text.splitlines()), "lines")


for pkg, src, fwd_name, model_fwd_name, init_sym, utils_mod in JOBS:
    src_path = os.path.join(REPO, pkg, "monkeypatch", src)
    text = open(src_path).read()
    prefix = fwd_name.rsplit("_llama_", 1)[0]

    # ---------------------------------------------------------------- models_ext
    body = extract(text, fwd_name)
    body, n = PROJ_RE.subn(PROJ_NEW, body)
    assert n == 1, f"{pkg}: projection block matched {n} times"
    rope_line = "    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)\n"
    assert body.count(rope_line) == 1, f"{pkg}: rope line not found exactly once"
    body = body.replace(rope_line, rope_line + SCALE_NEW)
    # `self.hidden_size` in the decode reshape assumes num_heads*head_dim == hidden_size;
    # -1 is equivalent where that holds and correct where it does not.
    body = body.replace("reshape(bsz, q_len, self.hidden_size)", "reshape(bsz, q_len, -1)")
    ext_fwd = f"{prefix}_ext_flash_attn2_forward"
    body = body.replace(f"def {fwd_name}(", f"def {ext_fwd}(")
    emit(
        os.path.join(REPO, pkg, "monkeypatch", f"{prefix}_ext_hijack.py"),
        HEADER.format(names=ext_fwd, plural="", pkg=pkg, src=src,
                      backbone="the models_ext backbones (Ministral3)"),
        ["from typing import Optional, Tuple", "",
         "import torch",
         "from transformers.cache_utils import Cache, StaticCache",
         "from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv",
         "from transformers.modeling_flash_attention_utils import _flash_attention_forward",
         "from transformers.utils import logging, is_flash_attn_2_available", "",
         f"from {pkg}.monkeypatch.{utils_mod} import {init_sym}", "",
         "logger = logging.get_logger(__name__)", "",
         "if is_flash_attn_2_available():",
         "    from flash_attn import flash_attn_varlen_func", "", ""],
        [body],
    )

    # -------------------------------------------------------------------- Qwen2
    attn = extract(text, fwd_name)
    attn = drop_position_embeddings_arg(attn)
    attn, n = ROPE_RE.subn(ROPE_QWEN2, attn)
    assert n == 1, f"{pkg}: qwen2 rope block matched {n} times"
    attn, n = SW_RE.subn(SW_QWEN2, attn)
    assert n == 1, f"{pkg}: sliding_window line matched {n} times"
    attn = attn.replace("reshape(bsz, q_len, self.hidden_size)", "reshape(bsz, q_len, -1)")
    q2_fwd = f"{prefix}_qwen2_flash_attn2_forward"
    attn = attn.replace(f"def {fwd_name}(", f"def {q2_fwd}(")

    model = extract(text, model_fwd_name)
    model, n = re.subn(r"[ \t]*position_embeddings = self\.rotary_emb\(hidden_states, position_ids\)\n",
                       "", model)
    assert n == 1, f"{pkg}: model rotary line matched {n} times"
    model, n = re.subn(r"[ \t]*position_embeddings(?:=position_embeddings)?,\n", "", model)
    assert n >= 1, f"{pkg}: model position_embeddings argument matched {n} times"
    anchor = "    if position_ids is None:\n        position_ids = cache_position.unsqueeze(0)\n"
    assert model.count(anchor) == 1, f"{pkg}: position_ids default block not found"
    model = model.replace(anchor, anchor + ROPE_STASH, 1)
    q2_model = f"{prefix}_qwen2_Qwen2Model_forward"
    model = model.replace(f"def {model_fwd_name}(", f"def {q2_model}(")

    emit(
        os.path.join(REPO, pkg, "monkeypatch", f"{prefix}_qwen2_hijack.py"),
        HEADER.format(names=f"{q2_model}, {q2_fwd}", plural="s", pkg=pkg, src=src,
                      backbone="Qwen2 (native in transformers 4.44)"),
        ["from typing import List, Optional, Tuple, Union", "",
         "import torch",
         # The LORE-KV model forward builds a plain DynamicCache for the prefill and
         # lookahead phases; the baselines' forwards do not.
         "from transformers.cache_utils import Cache, "
         + ("DynamicCache, " if pkg == "lorekv" else "") + "StaticCache",
         "from transformers.modeling_outputs import BaseModelOutputWithPast",
         "# Qwen2's own apply_rotary_pos_emb, which takes position_ids and indexes the",
         "# cos/sin table -- NOT Llama's, which expects already-gathered embeddings.",
         "from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv",
         "from transformers.modeling_flash_attention_utils import _flash_attention_forward",
         "from transformers.utils import logging, is_flash_attn_2_available", "",
         f"from {pkg}.monkeypatch.{utils_mod} import {init_sym}, DynamicCacheSplitHeadFlatten", "",
         "logger = logging.get_logger(__name__)", "",
         "if is_flash_attn_2_available():",
         "    from flash_attn import flash_attn_varlen_func", "", ""],
        [model, attn],
    )
