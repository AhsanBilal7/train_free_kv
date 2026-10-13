from importlib.metadata import version
import warnings
import transformers
import transformers.models.mistral.modeling_mistral
from andpro_snapkv.monkeypatch.andpro_mistral_hijack import andpro_mistral_flash_attn2_forward, andpro_MistralModel_forward
from andpro_snapkv.monkeypatch.andpro_mistral_hijack import prepare_inputs_for_generation_mistral as andpro_prepare_inputs_for_generation_mistral
from andpro_snapkv.monkeypatch.andpro_llama_hijack import andpro_llama_flash_attn2_forward, andpro_LlamaModel_forward
from andpro_snapkv.monkeypatch.andpro_llama_hijack import prepare_inputs_for_generation_llama as ada_prepare_inputs_for_generation_llama

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


def replace_mistral_andpro():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = andpro_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = andpro_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = andpro_MistralModel_forward

def replace_llama_andpro():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = andpro_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = andpro_LlamaModel_forward


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
# Guarded: models_ext reaches llama.tokenization_llama_fast, which transformers 5.x
# removed. pred.py imports this module unconditionally, so an unguarded ImportError
# here kills every run on the hybrid (5.x) venv -- including the hybrid backbones,
# which need none of this. On 4.44 the imports succeed and behaviour is unchanged;
# on 5.x the names stay undefined and the replace_* functions below are never called
# for these backbones (pred.py routes them to replace_hybrid_*).
try:
    from models_ext.hijack_targets import apply_ext_hijack
    from andpro_snapkv.monkeypatch.andpro_ext_hijack import andpro_ext_flash_attn2_forward
    from andpro_snapkv.monkeypatch.andpro_qwen2_hijack import (
        andpro_qwen2_flash_attn2_forward,
        andpro_qwen2_Qwen2Model_forward,
    )
except ImportError:      # transformers 5.x: 4.44-only hijacks unavailable
    pass


def replace_ministral3_andpro():
    check_version()
    apply_ext_hijack("ministral3", andpro_ext_flash_attn2_forward,
                     andpro_LlamaModel_forward, ada_prepare_inputs_for_generation_llama)


def replace_qwen2_andpro():
    check_version()
    apply_ext_hijack("qwen2", andpro_qwen2_flash_attn2_forward,
                     andpro_qwen2_Qwen2Model_forward, ada_prepare_inputs_for_generation_llama)
