"""LORE-KV: Lookahead Output-perturbation with Reliability-weighted Ensembles for KV caches.

A training-free KV-cache eviction method. Before the prompt's cache is compressed,
LORE-KV samples ``M`` short continuations of length ``T`` from the frozen model,
uses their query states as Monte Carlo samples of the queries the model will issue
while answering, weights each trajectory by its reliability, scores every cached
token by the projected change in attention output its deletion would cause, and
keeps the top-scoring tokens of every KV head. The continuations are discarded and
decoding uses only the compressed cache.

Subpackages and modules:

* :mod:`lorekv.monkeypatch` -- the method: phase-aware attention forwards for each
  backbone (``causal_*_hijack``), the scoring and selection
  (:mod:`~lorekv.monkeypatch.scoring`), and the generation loop
  (:mod:`~lorekv.monkeypatch.lookahead_runner`).
* :mod:`lorekv.harness` -- method selection, configuration and model loading
  shared by the benchmark drivers in ``experiments/``.
* :mod:`lorekv.data` -- LongBench and RULER loaders.
"""

__version__ = "1.0.0"
