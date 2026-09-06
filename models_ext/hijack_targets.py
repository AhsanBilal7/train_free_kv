"""The three class attributes every eviction package has to patch, per backbone.

Each `<pkg>/monkeypatch/monkeypatch.py` patches the same trio for Mistral and
Llama:

    <Model>ForCausalLM.prepare_inputs_for_generation
    <Model>FlashAttention2.forward
    <Model>Model.forward

For the models_ext backbones only the attention forward is package-specific --
they are Llama subclasses, so the Llama model forward and the Llama
`prepare_inputs_for_generation` apply verbatim. `apply_ext_hijack` below takes
those three callables and installs them, so a package's replace_* function stays
a two-liner and no package has to know how the vendored classes are built.
"""

from models_ext.modeling_ministral3 import (
    Ministral3FlashAttention2,
    Ministral3ForCausalLM,
    Ministral3Model,
)
from models_ext.modeling_qwen3 import Qwen3FlashAttention2, Qwen3ForCausalLM, Qwen3Model

# backbone name -> (attention class, model class, causal-LM class)
EXT_BACKBONES = {
    "qwen3": (Qwen3FlashAttention2, Qwen3Model, Qwen3ForCausalLM),
    "ministral3": (Ministral3FlashAttention2, Ministral3Model, Ministral3ForCausalLM),
}


def apply_ext_hijack(backbone, attn_forward, model_forward, prepare_inputs):
    """Install one package's hijack onto one vendored backbone.

    `model_forward` and `prepare_inputs` are the package's LLAMA functions: the
    vendored classes inherit Llama's model body and generation input prep
    unchanged, and setting them explicitly (rather than relying on inheriting
    whatever was patched onto LlamaModel) keeps the two backbones independent of
    the order in which pred.py calls the replace_* functions.
    """
    try:
        attn_cls, model_cls, causal_lm_cls = EXT_BACKBONES[backbone]
    except KeyError:
        raise ValueError(f"unknown models_ext backbone {backbone!r}; have {sorted(EXT_BACKBONES)}") from None
    causal_lm_cls.prepare_inputs_for_generation = prepare_inputs
    attn_cls.forward = attn_forward
    model_cls.forward = model_forward
