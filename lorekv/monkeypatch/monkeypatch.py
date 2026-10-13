"""Functions that install LORE-KV on a backbone.

``replace_{mistral,llama,qwen2,ministral3}_lookahead`` patch the dense backbones
(transformers 4.44). ``replace_hybrid_lookahead`` patches Qwen3.5 and Gemma 4
(transformers 5.x). :func:`lorekv.harness.install_method` calls these functions.
"""
from importlib.metadata import version
import warnings
import transformers
import transformers.models.mistral.modeling_mistral
from lorekv.monkeypatch.causal_mistral_hijack import (
    causal_lookahead_mistral_flash_attn2_forward,
    causal_lookahead_MistralModel_forward,
    prepare_inputs_for_generation_lookahead_mistral,
)
from lorekv.monkeypatch.causal_llama_hijack import (
    causal_lookahead_llama_flash_attn2_forward,
    causal_lookahead_LlamaModel_forward,
    prepare_inputs_for_generation_lookahead_llama,
)


def check_version():
    """Warn if the installed transformers is not release 4.44."""
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
        return
    if not transformers_version.startswith("4.44"):
        warnings.warn(f"transformers {transformers_version} is untested with these patches; "
                      "they target transformers 4.44.2 (see pyproject.toml).")


# config.lookahead_phase ("collect", "lookahead" or "final") selects the operation of
# the patched forwards. lorekv.monkeypatch.lookahead_runner sets the phase.
def replace_mistral_lookahead():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_lookahead_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = causal_lookahead_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = causal_lookahead_MistralModel_forward


def replace_llama_lookahead():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_lookahead_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = causal_lookahead_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = causal_lookahead_LlamaModel_forward


# ---------------------------------------------------------------------------
# Additional dense backbones.
#
# models_ext/ implements Ministral3 as a Llama subclass. It uses the Llama model
# forward and prepare_inputs_for_generation of this package. It needs its own
# attention forward, because its attention class defines `forward`. Qwen2 in
# transformers 4.44 uses an older rotary interface, so it also needs its own model
# forward. scripts/gen_ext_hijacks.py generates both files from the Llama patch.
# ---------------------------------------------------------------------------
# These imports are local. models_ext and the 4.44 patches use modules that
# transformers 5.x removed. A module-level import would prevent the use of this
# file in the 5.x environment of the hybrid backbones.


def replace_ministral3_lookahead():
    check_version()
    from models_ext.hijack_targets import apply_ext_hijack
    from lorekv.monkeypatch.causal_lookahead_ext_hijack import (
        causal_lookahead_ext_flash_attn2_forward)
    apply_ext_hijack("ministral3", causal_lookahead_ext_flash_attn2_forward,
                     causal_lookahead_LlamaModel_forward, prepare_inputs_for_generation_lookahead_llama)


def replace_qwen2_lookahead():
    check_version()
    from models_ext.hijack_targets import apply_ext_hijack
    from lorekv.monkeypatch.causal_lookahead_qwen2_hijack import (
        causal_lookahead_qwen2_flash_attn2_forward,
        causal_lookahead_qwen2_Qwen2Model_forward,
    )
    apply_ext_hijack("qwen2", causal_lookahead_qwen2_flash_attn2_forward,
                     causal_lookahead_qwen2_Qwen2Model_forward, prepare_inputs_for_generation_lookahead_llama)


# ---------------------------------------------------------------------------
# Hybrid-attention backbones (transformers 5.x): Qwen3.5, Gemma 4.
#
# These backbones run on transformers 5.x in a separate environment. Only the
# attention forward of the compressible layers is replaced. check_version() is not
# called, because it expects transformers 4.44.
# ---------------------------------------------------------------------------
def _patch_hybrid(attn_cls, model_cls, attn_forward):
    from lorekv.monkeypatch.causal_lookahead_hybrid_hijack import (
        make_hybrid_model_forward)
    # Keep the original forward for the layers that are not compressed.
    if not hasattr(attn_cls, "_lorekv_orig_forward"):
        attn_cls._lorekv_orig_forward = attn_cls.forward
    attn_cls.forward = attn_forward
    if not getattr(model_cls.forward, "_lorekv_wrapped", False):
        model_cls.forward = make_hybrid_model_forward(model_cls.forward)


def replace_qwen3_5_lookahead():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from lorekv.monkeypatch.causal_lookahead_hybrid_hijack import (
        causal_lookahead_qwen3_5_attention_forward)
    _patch_hybrid(M.Qwen3_5Attention, M.Qwen3_5TextModel if hasattr(M, "Qwen3_5TextModel") else M.Qwen3_5Model,
                  causal_lookahead_qwen3_5_attention_forward)


def replace_gemma4_lookahead():
    from transformers.models.gemma4_unified import modeling_gemma4_unified as M
    from lorekv.monkeypatch.causal_lookahead_hybrid_hijack import (
        causal_lookahead_gemma4_attention_forward)
    _patch_hybrid(M.Gemma4UnifiedTextAttention, M.Gemma4UnifiedTextModel,
                  causal_lookahead_gemma4_attention_forward)


def replace_hybrid_lookahead():
    """Patch the hybrid backbones that the installed transformers contains.

    The function does nothing on transformers 4.44, which does not contain these
    architectures. It catches only ImportError and AttributeError. Other errors
    propagate, because an unpatched model decodes with the full cache and gives an
    incorrect result for a compressed setting.
    """
    for name, fn in (("qwen3_5", replace_qwen3_5_lookahead),
                     ("gemma4_unified", replace_gemma4_lookahead)):
        try:
            fn()
        except (ImportError, ModuleNotFoundError, AttributeError):
            continue   # this transformers build has no such architecture
