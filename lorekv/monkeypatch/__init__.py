"""🔌 LORE-KV's monkeypatches: attention hijacks, scoring/selection and the generation loop.

Entry points: ``replace_*_lookahead`` in :mod:`.monkeypatch` install the method on
a backbone, and :func:`.lookahead_runner.generate_with_lookahead` runs Algorithm 1
for one prompt.
"""
