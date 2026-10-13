"""The three class attributes that each eviction package patches, for each backbone.

Each ``<pkg>/monkeypatch/monkeypatch.py`` patches these attributes for Mistral and
Llama::

    <Model>ForCausalLM.prepare_inputs_for_generation
    <Model>FlashAttention2.forward
    <Model>Model.forward

:func:`apply_ext_hijack` patches the same attributes for the two additional
backbones:

    ministral3: A Llama subclass in models_ext. The package gives its Llama model
        forward and prepare_inputs_for_generation. Only the attention forward is
        specific to the package (see scripts/gen_ext_hijacks.py).

    qwen2: The classes come from transformers 4.44. The package also gives a
        Qwen2 model forward, because Qwen2 uses an older rotary interface. The
        Llama prepare_inputs_for_generation is used, because it is equivalent.
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
    """Install the forwards of one package on one additional backbone.

    The function sets the model forward explicitly. The result thus does not depend
    on the order of the ``replace_*`` calls.

    Raises:
        ValueError: If ``backbone`` is not known.
    """
    try:
        attn_cls, model_cls, causal_lm_cls = EXT_BACKBONES[backbone]
    except KeyError:
        raise ValueError(f"unknown extra backbone {backbone!r}; have {sorted(EXT_BACKBONES)}") from None
    causal_lm_cls.prepare_inputs_for_generation = prepare_inputs
    attn_cls.forward = attn_forward
    model_cls.forward = model_forward
