from importlib.metadata import version
import warnings
import transformers
import transformers.models.mistral.modeling_mistral
from adaptive_snapkv.monkeypatch.fixed_mistral_hijack import fixed_mistral_flash_attn2_forward, fixed_MistralModel_forward
from adaptive_snapkv.monkeypatch.fixed_mistral_hijack import prepare_inputs_for_generation_mistral as fixed_prepare_inputs_for_generation_mistral
from adaptive_snapkv.monkeypatch.adaptive_mistral_hijack import adaptive_mistral_flash_attn2_forward,adaptive_MistralModel_forward
from adaptive_snapkv.monkeypatch.adaptive_mistral_hijack import prepare_inputs_for_generation_mistral as ada_prepare_inputs_for_generation_mistral

from adaptive_snapkv.monkeypatch.fixed_llama_hijack import fixed_llama_flash_attn2_forward, fixed_LlamaModel_forward
from adaptive_snapkv.monkeypatch.fixed_llama_hijack import prepare_inputs_for_generation_llama as fixed_prepare_inputs_for_generation_llama
from adaptive_snapkv.monkeypatch.adaptive_llama_hijack import adaptive_llama_flash_attn2_forward,adaptive_LlamaModel_forward
from adaptive_snapkv.monkeypatch.adaptive_llama_hijack import prepare_inputs_for_generation_llama as ada_prepare_inputs_for_generation_llama

from adaptive_snapkv.monkeypatch.slm_llama_hijack import slm_llama_flash_attn2_forward, slm_LlamaModel_forward
from adaptive_snapkv.monkeypatch.slm_mistral_hijack import slm_mistral_flash_attn2_forward, slm_MistralModel_forward

def check_version():
    """Warn when transformers is not the 4.44 release these patches were written for."""
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
        return
    if not transformers_version.startswith("4.44"):
        warnings.warn(f"transformers {transformers_version} is untested with these patches; "
                      "they target transformers 4.44.2 (see pyproject.toml).")


def replace_mistral_fixed():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = fixed_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = fixed_MistralModel_forward

def replace_mistral_adaptive():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = adaptive_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward

def replace_llama_fixed():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = fixed_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = fixed_LlamaModel_forward

def replace_llama_adaptive():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = adaptive_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward


def replace_llama_slm():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = slm_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = slm_LlamaModel_forward

def replace_mistral_slm():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = slm_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = slm_MistralModel_forward


# ---------------------------------------------------------------------------
# Extra backbones beyond Mistral and Llama; see lorekv for the
# rationale. Both files are generated from the Llama hijack by
# scripts/gen_ext_hijacks.py.
#
# `adaptive` covers BOTH the `ada` and the `critical` modes -- pred.py routes
# critical through replace_*_adaptive and the criterion is selected at runtime
# from model.config.mode -- so one pair of functions serves both.
# ---------------------------------------------------------------------------
# Guarded: models_ext reaches llama.tokenization_llama_fast, which transformers 5.x
# removed. pred.py imports this module unconditionally, so an unguarded ImportError
# here kills every run on the hybrid (5.x) venv -- including the hybrid backbones,
# which need none of this. On 4.44 the imports succeed and behaviour is unchanged;
# on 5.x the names stay undefined and the replace_* functions below are never called
# for these backbones (pred.py routes them to replace_hybrid_*).
try:
    from models_ext.hijack_targets import apply_ext_hijack
    from adaptive_snapkv.monkeypatch.adaptive_ext_hijack import adaptive_ext_flash_attn2_forward
    from adaptive_snapkv.monkeypatch.adaptive_qwen2_hijack import (
        adaptive_qwen2_flash_attn2_forward,
        adaptive_qwen2_Qwen2Model_forward,
    )
except ImportError:      # transformers 5.x: 4.44-only hijacks unavailable
    pass


def replace_ministral3_adaptive():
    check_version()
    apply_ext_hijack("ministral3", adaptive_ext_flash_attn2_forward,
                     adaptive_LlamaModel_forward, ada_prepare_inputs_for_generation_llama)


def replace_qwen2_adaptive():
    check_version()
    apply_ext_hijack("qwen2", adaptive_qwen2_flash_attn2_forward,
                     adaptive_qwen2_Qwen2Model_forward, ada_prepare_inputs_for_generation_llama)
