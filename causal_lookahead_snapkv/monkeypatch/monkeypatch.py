from importlib.metadata import version
import warnings
import transformers
import transformers.models.mistral.modeling_mistral
from causal_lookahead_snapkv.monkeypatch.causal_mistral_hijack import (
    causal_lookahead_mistral_flash_attn2_forward,
    causal_lookahead_MistralModel_forward,
    prepare_inputs_for_generation_lookahead_mistral,
)
from causal_lookahead_snapkv.monkeypatch.causal_llama_hijack import (
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
# See causal_lookahead_snapkv.monkeypatch.lookahead_runner for the orchestration that flips it.
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
from models_ext.hijack_targets import apply_ext_hijack
from causal_lookahead_snapkv.monkeypatch.causal_lookahead_ext_hijack import causal_lookahead_ext_flash_attn2_forward
from causal_lookahead_snapkv.monkeypatch.causal_lookahead_qwen2_hijack import (
    causal_lookahead_qwen2_flash_attn2_forward,
    causal_lookahead_qwen2_Qwen2Model_forward,
)


def replace_ministral3_lookahead():
    check_version()
    apply_ext_hijack("ministral3", causal_lookahead_ext_flash_attn2_forward,
                     causal_lookahead_LlamaModel_forward, prepare_inputs_for_generation_lookahead_llama)


def replace_qwen2_lookahead():
    check_version()
    apply_ext_hijack("qwen2", causal_lookahead_qwen2_flash_attn2_forward,
                     causal_lookahead_qwen2_Qwen2Model_forward, prepare_inputs_for_generation_lookahead_llama)
