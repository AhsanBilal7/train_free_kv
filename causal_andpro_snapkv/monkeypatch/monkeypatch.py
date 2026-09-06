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


# --- models_ext backbones (Qwen3, Ministral3); see causal_lookahead_snapkv for the rationale ---
from models_ext.hijack_targets import apply_ext_hijack
from causal_andpro_snapkv.monkeypatch.causal_ext_hijack import causal_ext_flash_attn2_forward


def replace_qwen3_causal():
    check_version()
    apply_ext_hijack("qwen3", causal_ext_flash_attn2_forward,
                     causal_LlamaModel_forward, causal_prepare_inputs_for_generation_llama)


def replace_ministral3_causal():
    check_version()
    apply_ext_hijack("ministral3", causal_ext_flash_attn2_forward,
                     causal_LlamaModel_forward, causal_prepare_inputs_for_generation_llama)
