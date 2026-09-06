"""Ministral3 (text tower) configuration for transformers 4.44.

`mistralai/Ministral-3-8B-Instruct-2512` ships as a `Mistral3ForConditionalGeneration`
checkpoint: a Pixtral vision tower plus a `ministral3` text model. Only the text
tower is used here, and scripts/convert_ministral3.py writes it out as a
standalone checkpoint whose config.json is exactly what this class parses.

Two fields drive the departures from Llama, both carried inside `rope_parameters`
verbatim from upstream so nothing is re-derived:
  * a YaRN inverse-frequency schedule (`type: yarn`, factor 16 over an original
    16384-token window), and
  * `llama_4_scaling_beta`, a post-RoPE query scale that grows with position.
"""

from transformers.configuration_utils import PretrainedConfig

# Upstream Ministral3Config.__post_init__ defaults, used when a checkpoint omits
# rope_parameters. Kept as a literal so this file needs no transformers 5.x import.
DEFAULT_ROPE_PARAMETERS = {
    "type": "yarn",
    "rope_type": "yarn",
    "rope_theta": 1000000.0,
    "factor": 16.0,
    "original_max_position_embeddings": 16384,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "llama_4_scaling_beta": 0.1,
}


class Ministral3Config(PretrainedConfig):
    model_type = "ministral3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        sliding_window=None,
        attention_dropout=0.0,
        attention_bias=False,
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
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        rp = dict(DEFAULT_ROPE_PARAMETERS)
        if rope_parameters:
            rp.update(rope_parameters)
        rp.setdefault("rope_type", rp.get("type", "yarn"))
        self.rope_parameters = rp

        # Llama-derived modules read these two directly. `rope_scaling = None`
        # keeps `LlamaRotaryEmbedding` on its default branch -- harmless, because
        # Ministral3Attention replaces `self.rotary_emb` with the YaRN one, and
        # deliberate, because 4.44's own "yarn" branch uses a DIFFERENT correction
        # range (max_position_embeddings, not the original 16384 window) and a
        # different attention factor than the release this checkpoint targets.
        self.rope_theta = rp["rope_theta"]
        self.rope_scaling = None
        # Read by Llama's modules: no tensor-parallel weight splitting, and the
        # SwiGLU projections are bias-free.
        self.pretraining_tp = 1
        self.mlp_bias = False

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
