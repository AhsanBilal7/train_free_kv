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
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
        return
    version_list = ['4.37']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


# NOTE: the active phase (model.model.config.lookahead_phase, one of "collect"/"lookahead"/"final")
# is what decides at runtime whether the patched Model.forward/attention forward run plain
# full-attention (prefill + lookahead-particle decode) or the post-eviction varlen decode path.
# See lorekv.monkeypatch.lookahead_runner for the orchestration that flips it.
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
# Extra backbones beyond Mistral and Llama.
#
# Ministral3 is vendored in models_ext/ as a Llama subclass, so it reuses this
# package's Llama model forward and prepare_inputs_for_generation unchanged; only
# the attention forward is re-emitted, because the vendored attention class
# defines its own `forward` and would otherwise shadow the LlamaFlashAttention2
# patch. Qwen2 is native in transformers 4.44 but predates the rotary refactor,
# so it brings its own model forward too. Both files are generated from the Llama
# hijack by scripts/gen_ext_hijacks.py.
# ---------------------------------------------------------------------------
# Imported lazily. models_ext and the 4.44 hijacks reach into modules that
# transformers 5.x removed (e.g. llama.tokenization_llama_fast), so importing them
# at module scope would make this whole file unimportable on the 5.x venv the
# hybrid backbones run in -- taking replace_qwen3_5_lookahead down with it.


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
# These are NOT vendored into models_ext/ the way Ministral3 was. Ministral3 is
# "Llama plus a tweak", so a 4.44-compatible subclass was a few hundred lines.
# Qwen3.5 carries a whole second mixer (Qwen3_5GatedDeltaNet) and Gemma 4 is a
# unified multimodal stack; back-porting either to 4.44 would mean re-implementing
# machinery we never touch. Instead they run on transformers 5.x in a separate
# venv, and only the attention forward of the *compressible* layers is replaced.
#
# check_version() is deliberately NOT called: it warns for anything that is not
# 4.37 and these run on 5.16 by design.
# ---------------------------------------------------------------------------
def _patch_hybrid(attn_cls, model_cls, attn_forward):
    from lorekv.monkeypatch.causal_lookahead_hybrid_hijack import (
        make_hybrid_model_forward)
    # Keep the original bound so a non-compressible layer can fall through to it.
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
    """Patch whichever hybrid backbones this transformers build actually has.

    pred.py calls every replace_* unconditionally, so this must be a no-op rather
    than an error on the 4.44 venv, where transformers.models.qwen3_5 does not
    exist at all. Guarded per model so a future partial install still patches what
    it can. Anything unexpected is re-raised: a silently unpatched model would run
    uncompressed and report a full-cache score under a compression label, which is
    the worst possible failure here.
    """
    for name, fn in (("qwen3_5", replace_qwen3_5_lookahead),
                     ("gemma4_unified", replace_gemma4_lookahead)):
        try:
            fn()
        except (ImportError, ModuleNotFoundError, AttributeError):
            continue   # this transformers build has no such architecture
