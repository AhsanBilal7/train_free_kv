"""LORE-KV: Lookahead Output-perturbation with Reliability-weighted Ensembles for KV caches.

LORE-KV is a training-free method for KV-cache eviction. It samples M short
continuations of T tokens from the frozen model. It uses their query states as
Monte Carlo samples of the queries that the model issues while it answers. It
weights each continuation by its reliability. It scores each cached token by the
projected change of the attention output when the token is deleted. Each KV head
keeps its tokens with the highest scores. The method then discards the
continuations and decodes from the compressed cache.

Modules:
    lorekv.monkeypatch: The method. It contains the attention forwards for each
        backbone, the scoring and selection, and the generation loop.
    lorekv.harness: Method selection, configuration and model loading for the
        benchmark drivers in ``experiments/``.
    lorekv.data: Loaders for LongBench and RULER.
"""

__version__ = "1.0.0"
