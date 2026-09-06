"""Backbone support that the pinned Transformers release does not ship.

The repo is pinned to transformers 4.44 because every eviction hijack in
adaptive_snapkv / andpro_snapkv / causal_andpro_snapkv / causal_lookahead_snapkv
patches `*FlashAttention2.forward` and `*Model.forward`, classes that were
removed in 4.48 when attention moved behind `ALL_ATTENTION_FUNCTIONS`. Two of
the backbones the paper reports were released after that pin:

    Qwen/Qwen3-8B                            -> `qwen3`      (transformers >= 4.51)
    mistralai/Ministral-3-8B-Instruct-2512   -> `ministral3` (transformers  5.x)

Upgrading transformers to reach them would invalidate every hijack, so instead
this package re-expresses the two architectures in the 4.44 Llama idiom. Both
model files subclass `transformers.models.llama.modeling_llama`, so they expose
exactly the surface the hijacks patch (`*FlashAttention2`, `*Model`,
`*ForCausalLM.prepare_inputs_for_generation`) and inherit Llama's cache,
mask and rotary plumbing unchanged.

Correctness is not assumed: scripts/validate_models_ext.py loads the same
checkpoint under upstream transformers 5.x in a separate venv and asserts the
logits agree. See README.md in this directory for the recorded results.
"""

from models_ext.register import register_extra_architectures

__all__ = ["register_extra_architectures"]
