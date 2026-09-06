"""The three class attributes every eviction package has to patch, per backbone.

Each `<pkg>/monkeypatch/monkeypatch.py` patches the same trio for Mistral and
Llama:

    <Model>ForCausalLM.prepare_inputs_for_generation
    <Model>FlashAttention2.forward
    <Model>Model.forward

`apply_ext_hijack` installs that trio for the two backbones added on top of the
pinned Transformers release, so a package's replace_* function stays a two-liner
and no package has to know how either backbone is put together:

  ministral3  vendored in models_ext as a Llama subclass, so the package passes
              its LLAMA model forward and prepare_inputs verbatim; only the
              attention forward is package-specific (see gen_ext_hijacks.py).

  qwen2       native in transformers 4.44, so the classes come straight from
              transformers. Its 4.44 implementation predates the rotary refactor,
              so the package passes a Qwen2-specific model forward as well;
              prepare_inputs_for_generation is byte-identical to Llama's apart
              from a StaticCache branch these hijacks reject anyway, so the
              package's Llama one is reused.
"""

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2FlashAttention2,
    Qwen2ForCausalLM,
    Qwen2Model,
)

from models_ext.modeling_ministral3 import (
    Ministral3FlashAttention2,
    Ministral3ForCausalLM,
    Ministral3Model,
)

# backbone name -> (attention class, model class, causal-LM class)
EXT_BACKBONES = {
    "ministral3": (Ministral3FlashAttention2, Ministral3Model, Ministral3ForCausalLM),
    "qwen2": (Qwen2FlashAttention2, Qwen2Model, Qwen2ForCausalLM),
}


def apply_ext_hijack(backbone, attn_forward, model_forward, prepare_inputs):
    """Install one package's hijack onto one extra backbone.

    Setting the model forward explicitly (rather than relying on Ministral3
    inheriting whatever was patched onto LlamaModel) keeps the backbones
    independent of the order in which pred.py calls the replace_* functions.
    """
    try:
        attn_cls, model_cls, causal_lm_cls = EXT_BACKBONES[backbone]
    except KeyError:
        raise ValueError(f"unknown extra backbone {backbone!r}; have {sorted(EXT_BACKBONES)}") from None
    causal_lm_cls.prepare_inputs_for_generation = prepare_inputs
    attn_cls.forward = attn_forward
    model_cls.forward = model_forward
