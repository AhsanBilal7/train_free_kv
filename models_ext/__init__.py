"""Backbones that the pinned transformers release does not contain.

The dense-backbone experiments use transformers 4.44. Each eviction patch in
adaptive_snapkv, andpro_snapkv and lorekv replaces ``*FlashAttention2.forward``
and ``*Model.forward``, and release 4.48 removed these classes.

Two backbones need additional support:

    mistralai/Ministral-3-8B-Instruct-2512 (``ministral3``): The architecture is
        available only in transformers 5.x. ``modeling_ministral3.py`` implements
        it as a subclass of the 4.44 Llama modules. It thus has the same interface
        as Llama for the patches.

    Qwen/Qwen2.5-*-Instruct (``qwen2``): The architecture is in transformers 4.44,
        but it uses an older rotary interface. It thus needs its own patched
        forwards. ``scripts/gen_ext_hijacks.py`` generates them, and
        ``hijack_targets.py`` installs them.

``scripts/validate_models_ext.py`` compares the logits of this implementation
with the transformers 5.x implementation. With uncompressed fp32 inference the
Ministral3 implementation gives the same logits and top-1 predictions (paper
App. E.3).
"""

# models_ext vendors Ministral-3 as a subclass of transformers 4.44's Llama modules
# (it needs LlamaFlashAttention2, which 4.48 deleted). That makes the package
# unimportable on the transformers 5.x venv used by the hybrid-attention backbones
# -- and every experiments/*/pred.py imports it unconditionally, so an unguarded
# ImportError there kills runs that never wanted Ministral-3 in the first place.
#
# On 5.x, registration degrades to a no-op: the extra architectures it registers are
# 4.44-only backbones, and the 5.x-native ones (qwen3_5, gemma4_unified) are already
# known to AutoConfig without any help. The reason is recorded on the function so a
# caller can tell "registered nothing because nothing was needed" from "silently
# broken".
try:
    from models_ext.register import register_extra_architectures
    register_extra_architectures.unavailable_reason = None
except ImportError as _exc:                      # transformers 5.x: no LlamaFlashAttention2
    _reason = str(_exc)

    def register_extra_architectures():          # type: ignore[misc]
        """No-op: the vendored 4.44-only architectures cannot load on this stack."""
        return None

    register_extra_architectures.unavailable_reason = _reason

__all__ = ["register_extra_architectures"]
