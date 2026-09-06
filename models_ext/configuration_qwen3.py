"""Qwen3 configuration for transformers 4.44.

Field-for-field the subset of upstream `Qwen3Config` that `Qwen/Qwen3-8B`'s
config.json actually sets, expressed as a 4.44 `PretrainedConfig` so that the
Llama-derived modules in modeling_qwen3.py can read it. Qwen3's only structural
departures from Llama are per-head QK RMSNorm and an explicit `head_dim`; both
are carried here.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Qwen3Config(PretrainedConfig):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=36,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # `use_sliding_window` gates `sliding_window` upstream; Qwen3-8B ships it
        # False, so attention is dense at every layer. Mirror that gating here so
        # a checkpoint that does turn it on is not silently run dense.
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        # Llama's modules read `pretraining_tp` and `mlp_bias`; Qwen3 has no
        # tensor-parallel weight splitting (1 is the "no slicing" path) and its
        # SwiGLU projections are bias-free, like Llama-3's.
        self.pretraining_tp = 1
        self.mlp_bias = False

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
