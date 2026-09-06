"""Register the vendored backbones with the Auto* factories.

pred.py loads every backbone through `AutoModelForCausalLM.from_pretrained`, so
`qwen3` and `ministral3` only need to be findable by model_type. Calling this is
idempotent and a no-op for any model_type transformers already knows, so it is
safe to call unconditionally at import time.
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from models_ext.configuration_ministral3 import Ministral3Config
from models_ext.configuration_qwen3 import Qwen3Config
from models_ext.modeling_ministral3 import Ministral3ForCausalLM
from models_ext.modeling_qwen3 import Qwen3ForCausalLM

# Both checkpoints name their tokenizer in tokenizer_config.json, so AutoTokenizer
# normally never needs this mapping. It does need it on a compute node: ACES workers
# have no outbound network, the tokenizer_config fetch fails, `config_tokenizer_class`
# comes back None, and AutoTokenizer falls through to TOKENIZER_MAPPING[type(config)]
# -- which without this raises KeyError('Qwen3Config') at model-load time.
_TOKENIZERS = {
    "qwen3": Qwen2TokenizerFast,       # Qwen3 reuses the Qwen2 BPE tokenizer
    "ministral3": LlamaTokenizerFast,  # tekken, exported in the Llama fast format
}

_REGISTERED = False


def register_extra_architectures():
    global _REGISTERED
    if _REGISTERED:
        return
    for config_cls, model_cls in ((Qwen3Config, Qwen3ForCausalLM), (Ministral3Config, Ministral3ForCausalLM)):
        try:
            AutoConfig.register(config_cls.model_type, config_cls)
            AutoModelForCausalLM.register(config_cls, model_cls)
            AutoTokenizer.register(config_cls, fast_tokenizer_class=_TOKENIZERS[config_cls.model_type])
        except ValueError:
            # Already registered -- either by a second call in the same process or
            # by a transformers release that has since shipped the architecture.
            # In the latter case upstream wins, which is the outcome we want.
            pass
    _REGISTERED = True
