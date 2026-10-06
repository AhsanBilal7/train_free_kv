"""Register the vendored backbone with the Auto* factories.

Only Ministral3 is vendored: `qwen2` and every other backbone the paper uses is
native to the pinned Transformers release. pred.py loads every backbone through
`AutoModelForCausalLM.from_pretrained`, so `ministral3` only needs to be findable
by model_type. Calling this is idempotent and a no-op for any model_type
transformers already knows, so it is safe to call unconditionally at import time.
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# Import from the package root, not transformers.models.llama.tokenization_llama_fast:
# transformers 5.x removed that submodule path, and models_ext is imported
# unconditionally by every experiments/*/pred.py -- so the deep path made the whole
# run unimportable on the hybrid (5.x) venv. The root re-export exists in both 4.44
# and 5.x.
from transformers import LlamaTokenizerFast

from models_ext.configuration_ministral3 import Ministral3Config
from models_ext.modeling_ministral3 import Ministral3ForCausalLM

# Ministral-3 names its tokenizer in tokenizer_config.json, so AutoTokenizer
# normally never needs this mapping. It does need it on a compute node: ACES
# workers have no outbound network, the tokenizer_config fetch fails,
# `config_tokenizer_class` comes back None, and AutoTokenizer falls through to
# TOKENIZER_MAPPING[type(config)] -- which without this raises
# KeyError('Ministral3Config') at model-load time.
_TOKENIZERS = {
    "ministral3": LlamaTokenizerFast,  # tekken, exported in the Llama fast format
}

_REGISTERED = False


def register_extra_architectures():
    global _REGISTERED
    if _REGISTERED:
        return
    for config_cls, model_cls in ((Ministral3Config, Ministral3ForCausalLM),):
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
