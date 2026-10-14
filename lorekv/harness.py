"""Benchmark harness for ``experiments/{LongBench,RULER,NIAH}/pred.py``.

Each benchmark driver uses the same four steps. All methods thus use the same
budgets, observation window, chunk size and selector (paper Sec. 4, App. E.1).

1. :func:`add_method_args` adds the arguments that select the eviction method and
   its hyperparameters (paper Table 4).
2. :func:`install_method` patches the attention and model forwards of each
   supported backbone. Call it before you load the model.
3. :func:`load_model_and_tokenizer` loads the frozen backbone in bfloat16.
4. :func:`configure_model` writes the hyperparameters to the model config. The
   patched attention layers read them from there.

Methods (``--mode``):

========================  ====================================================
``causal_lookahead``      LORE-KV (this work), :mod:`lorekv.monkeypatch`
``andpro``                AnDPro (Geng et al., NeurIPS 2025), :mod:`andpro_snapkv`
``critical``              CriticalKV (Feng et al., ICML 2025) on Ada-KV head
                          budgets, :mod:`adaptive_snapkv`
``ada``                   Ada-KV (Feng et al., 2025), :mod:`adaptive_snapkv`
``fix``                   SnapKV (Li et al., 2024); ``--pyram`` gives PyramidKV
``slm``                   StreamingLLM (Xiao et al., 2024)
``full``                  Uncompressed full-cache reference
========================  ====================================================

The ``ada``, ``fix``, ``slm`` and ``critical`` patches and their per-head cache
come from AdaKV (https://github.com/FFY0/AdaKV, MIT). ``critical`` adds the
two-stage CriticalKV selection, which scores each entry by its attention and the
L1 norm of its value after the output projection (see
:func:`adaptive_snapkv.monkeypatch.snapkv_utils.criticalkv_select`). The ``andpro`` patches come
from LLM-AnDPro (https://github.com/MIRALab-USTC/LLM-AnDPro, MIT). This
repository is built on LLM-AnDPro.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

METHODS = ("causal_lookahead", "andpro", "critical", "ada", "fix", "slm", "full")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def add_method_args(parser):
    """Add the eviction-method arguments to a parser.

    The LORE-KV defaults are the core configuration of the paper, LORE-KV /E
    (Table 4): w = 32, M = 4, T = 8, lambda_L = 1, (beta, gamma, lambda_D) =
    (1, 1.3, 0) and chunk size 4. All plug-ins are off.

    Args:
        parser: An ``argparse.ArgumentParser``.

    Returns:
        The same parser.
    """
    g = parser.add_argument_group("eviction method")
    g.add_argument("--mode", type=str, default="causal_lookahead", choices=METHODS,
                   help="KV-eviction method; causal_lookahead is LORE-KV")
    g.add_argument("--budget", default=128, type=int,
                   help="cache budget B: KV pairs retained per KV head per layer")
    g.add_argument("--chunk_size", default=4, type=int,
                   help="adjacent tokens grouped into one selection unit (SnapKV/Ada-KV)")
    g.add_argument("--floor_alpha", default=0.2, type=float,
                   help="[ada/critical] fraction of each head's budget that is not adaptive")
    g.add_argument("--gqa_support", action="store_true",
                   help="[ada/critical] score at KV-head granularity under GQA")
    g.add_argument("--gqa_func", default="mean", choices=["mean", "max"],
                   help="[ada/critical] how query heads in a GQA group are pooled")
    g.add_argument("--pyram", action="store_true",
                   help="[fix/ada] PyramidKV layer-wise budget schedule")
    g.add_argument("--pyram_beta", default=20, type=int,
                   help="[fix/ada] PyramidKV min/max budget ratio")

    g = parser.add_argument_group("LORE-KV (--mode causal_lookahead)")
    g.add_argument("--num_particles", default=4, type=int,
                   help="M: number of sampled lookahead trajectories (Eq. 1); 0 = prompt window only")
    g.add_argument("--lookahead_len", default=8, type=int,
                   help="T: decoding steps per lookahead trajectory (Eq. 1)")
    g.add_argument("--lookahead_temperature", default=1.0, type=float,
                   help="sampling temperature of the trajectories; 1.0 keeps Eq. 2 exactly centered")
    g.add_argument("--beta", default=1.0, type=float,
                   help="beta: coefficient of the centered typicality c_bar in Eq. 3")
    g.add_argument("--gamma", default=1.3, type=float,
                   help="gamma: coefficient of the mean entropy H_bar in Eq. 3")
    g.add_argument("--lambda_d", default=0.0, type=float,
                   help="lambda_D: coefficient of the query-space disagreement D_hat in Eq. 3/10")
    g.add_argument("--normalize_typicality", default=0, type=int, choices=[0, 1],
                   help="use c_bar / sqrt(V_bar/T) (App. C.3); ablation only")
    g.add_argument("--lambda_lookahead", default=1.0, type=float,
                   help="lambda_L: weight of the lookahead term in Eq. 7; 0 = prompt window only")
    g.add_argument("--robust_alpha", default=0.0, type=float,
                   help="alpha: TopAvg/Mean mixing of the aggregation psi (App. D-A); 0 = mean")
    g.add_argument("--robust_topk_beta", default=1.0, type=float,
                   help="kappa: fraction of queries kept by TopAvg (App. D-A)")
    g.add_argument("--budget_tau", default=0.0, type=float,
                   help="tau: adaptive head-budget exponent (Eq. 11); 0 = uniform")
    g.add_argument("--lambda_k", default=0.0, type=float,
                   help="lambda_k: key-redundancy penalty of diversity selection (App. D-C)")
    g.add_argument("--lambda_v", default=0.0, type=float,
                   help="lambda_v: value-redundancy penalty of diversity selection (App. D-C)")
    g.add_argument("--score_objective", default="projected",
                   choices=["projected", "unprojected", "attention", "andpro"],
                   help="per-query deletion score: ||W_O dO|| (Eq. 6), ||dO||, attention "
                        "mass or AnDPro's criterion (App. D, ablation iii; Table 13b)")
    return parser


# ---------------------------------------------------------------------------
# Method installation
# ---------------------------------------------------------------------------
def _has_legacy_flash_attn_stack():
    """Return True if transformers contains the ``*FlashAttention2`` classes (release 4.44).

    The dense-backbone patches replace these classes. Release 4.48 removed them. The
    hybrid backbones run on transformers 5.x and use only the hybrid patches.
    """
    import transformers.models.mistral.modeling_mistral as m
    return hasattr(m, "MistralFlashAttention2")


def install_method(mode):
    """Patch each supported backbone with the method ``mode``.

    Call this function before you load the model.

    Args:
        mode: One of :data:`METHODS`.

    Raises:
        ValueError: If ``mode`` is not a known method.
    """
    if mode == "full":
        return
    legacy = _has_legacy_flash_attn_stack()

    from lorekv.monkeypatch.hybrid_baseline_hijack import replace_hybrid_baseline

    if mode == "causal_lookahead":
        from lorekv.monkeypatch.monkeypatch import (
            replace_hybrid_lookahead, replace_llama_lookahead, replace_ministral3_lookahead,
            replace_mistral_lookahead, replace_qwen2_lookahead)
        patches = (replace_mistral_lookahead, replace_llama_lookahead,
                   replace_qwen2_lookahead, replace_ministral3_lookahead)
        hybrid = replace_hybrid_lookahead
    elif mode == "andpro":
        from andpro_snapkv.monkeypatch.monkeypatch import (
            replace_llama_andpro, replace_ministral3_andpro, replace_mistral_andpro,
            replace_qwen2_andpro)
        patches = (replace_mistral_andpro, replace_llama_andpro,
                   replace_qwen2_andpro, replace_ministral3_andpro)
        hybrid = lambda: replace_hybrid_baseline("andpro")  # noqa: E731
    elif mode in ("ada", "critical"):
        from adaptive_snapkv.monkeypatch.monkeypatch import (
            replace_llama_adaptive, replace_ministral3_adaptive, replace_mistral_adaptive,
            replace_qwen2_adaptive)
        patches = (replace_mistral_adaptive, replace_llama_adaptive,
                   replace_qwen2_adaptive, replace_ministral3_adaptive)
        hybrid = lambda: replace_hybrid_baseline("adaptive")  # noqa: E731
    elif mode == "fix":
        from adaptive_snapkv.monkeypatch.monkeypatch import replace_llama_fixed, replace_mistral_fixed
        patches, hybrid = (replace_mistral_fixed, replace_llama_fixed), None
    elif mode == "slm":
        from adaptive_snapkv.monkeypatch.monkeypatch import replace_llama_slm, replace_mistral_slm
        patches, hybrid = (replace_mistral_slm, replace_llama_slm), None
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {METHODS}")

    if legacy:
        for fn in patches:
            fn()
    if hybrid is not None:
        hybrid()


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
def _config_chain(model):
    """Return each config object that a patched attention layer can read.

    A dense ``ForCausalLM`` model has one config object. A multimodal model (Gemma 4)
    also has a nested ``text_config``, and its attention layers read that object.
    """
    seen, out = set(), []
    for cfg in (getattr(model, "config", None),
                getattr(getattr(model, "model", None), "config", None)):
        while cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            out.append(cfg)
            cfg = getattr(cfg, "text_config", None)
    return out


def configure_model(model, args, window_size=32, kernel_size=7, pooling="maxpool", skip=0):
    """Write the method hyperparameters to each config in :func:`_config_chain`.

    ``window_size``, ``kernel_size`` and ``pooling`` set the SnapKV observation window
    and smoothing. All methods use the same values (paper App. E.1). The LORE-KV
    values are read by :func:`lorekv.monkeypatch.scoring.init_lookahead_snapkv` and
    :func:`lorekv.monkeypatch.lookahead_runner.generate_with_lookahead`.

    Args:
        model: The loaded model.
        args: The namespace from a parser built with :func:`add_method_args`.
        window_size: Observation window w.
        kernel_size: Pooling kernel of the SnapKV-family baselines.
        pooling: Pooling type of the SnapKV-family baselines.
        skip: Number of first layers that are not compressed.

    Returns:
        The same model.
    """
    fields = {
        # All methods
        "window_size": window_size, "base_capacity": args.budget, "kernel_size": kernel_size,
        "pooling": pooling, "skip": skip, "chunk_size": args.chunk_size, "mode": args.mode,
        # Ada-KV / PyramidKV
        "floor_alpha": args.floor_alpha, "normalize": None, "pyram_mode": args.pyram,
        "pyram_beta": args.pyram_beta, "gqa_support": args.gqa_support, "gqa_func": args.gqa_func,
        # LORE-KV
        "lookahead_phase": "collect", "lookahead_particle_idx": 0,
        "num_particles": args.num_particles, "lookahead_len": args.lookahead_len,
        "lookahead_temperature": args.lookahead_temperature,
        "particle_beta": args.beta, "particle_gamma": args.gamma, "lambda_d": args.lambda_d,
        "normalize_typicality": bool(args.normalize_typicality),
        "robust_alpha": args.robust_alpha, "robust_topk_beta": args.robust_topk_beta,
        "lambda_lookahead": args.lambda_lookahead, "lambda_k": args.lambda_k,
        "lambda_v": args.lambda_v, "budget_tau": args.budget_tau,
        "score_objective": args.score_objective,
    }
    for cfg in _config_chain(model):
        for name, value in fields.items():
            setattr(cfg, name, value)
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def attn_implementation(model_path=None):
    """Return ``"flash_attention_2"`` if it supports the model, else ``"sdpa"``.

    The function returns ``"sdpa"`` if flash-attn is not installed or if the head
    dimension is more than 192. Transformers limits flash-attn to a head dimension of
    192, and Qwen3.5 and Gemma 4 use 256. The choice does not change the method. The
    post-eviction decode step always calls ``flash_attn_varlen_func``.
    """
    try:
        import flash_attn
        if getattr(flash_attn, "__lorekv_stub__", False):
            return "sdpa"
    except Exception:
        return "sdpa"
    if model_path:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path)
            tcfg = getattr(cfg, "text_config", None) or cfg
            hd = tcfg.to_dict().get("head_dim")
            if isinstance(hd, (list, tuple)):
                hd = max(hd)
            if hd and int(hd) > 192:
                return "sdpa"
        except Exception:
            pass
    return "flash_attention_2"


def load_model_and_tokenizer(path, device="cuda:0"):
    """Load a frozen backbone in bfloat16 on one GPU.

    Call :func:`install_method` first.

    Args:
        path: A Hugging Face Hub ID or a local checkpoint directory.
        device: The target device.

    Returns:
        A tuple ``(model, tokenizer)``. The model is in evaluation mode.
    """
    # Register the Ministral3 implementation of models_ext with the Auto* classes.
    # This has no effect on the other backbones.
    from models_ext import register_extra_architectures
    register_extra_architectures()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if "gemma" in str(path).lower() and not getattr(tokenizer, "chat_template", None):
        # Gemma 4 is evaluated with its chat template (paper App. E.1). Without it the
        # drivers would send a raw prompt without <bos>.
        raise ValueError(
            f"{path} has no chat template. Copy chat_template.jinja from the Hub "
            "checkpoint into this directory (scripts/prepare_local_checkpoint.py copies it).")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation=attn_implementation(path), trust_remote_code=True)
    return model.eval(), tokenizer


def seed_everything(seed=42):
    """Seed the Python, NumPy and PyTorch random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
