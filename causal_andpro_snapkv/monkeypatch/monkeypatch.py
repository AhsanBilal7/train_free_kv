from importlib.metadata import version
import warnings
import transformers
import transformers.models.mistral.modeling_mistral
from causal_andpro_snapkv.monkeypatch.causal_mistral_hijack import causal_mistral_flash_attn2_forward, causal_MistralModel_forward
from causal_andpro_snapkv.monkeypatch.causal_mistral_hijack import prepare_inputs_for_generation_mistral as causal_prepare_inputs_for_generation_mistral
from causal_andpro_snapkv.monkeypatch.causal_llama_hijack import causal_llama_flash_attn2_forward, causal_LlamaModel_forward
from causal_andpro_snapkv.monkeypatch.causal_llama_hijack import prepare_inputs_for_generation_llama as causal_prepare_inputs_for_generation_llama

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.37']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


# NOTE: the mode string on the model config (set via config_compress in pred.py) decides at
# runtime whether AnDProAdaptiveSnapKVCluster.update_kv takes the causal_forward_only scoring
# path or falls back to the legacy AnDPro chunk-projection score.
def replace_mistral_causal():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = causal_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = causal_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = causal_MistralModel_forward

def replace_llama_causal():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = causal_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = causal_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = causal_LlamaModel_forward


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
    from causal_andpro_snapkv.monkeypatch.causal_ext_hijack import causal_ext_flash_attn2_forward
    from causal_andpro_snapkv.monkeypatch.causal_qwen2_hijack import (
        causal_qwen2_flash_attn2_forward,
        causal_qwen2_Qwen2Model_forward,
    )
except ImportError:      # transformers 5.x: 4.44-only hijacks unavailable
    pass


def replace_ministral3_causal():
    check_version()
    apply_ext_hijack("ministral3", causal_ext_flash_attn2_forward,
                     causal_LlamaModel_forward, causal_prepare_inputs_for_generation_llama)


def replace_qwen2_causal():
    check_version()
    apply_ext_hijack("qwen2", causal_qwen2_flash_attn2_forward,
                     causal_qwen2_Qwen2Model_forward, causal_prepare_inputs_for_generation_llama)
