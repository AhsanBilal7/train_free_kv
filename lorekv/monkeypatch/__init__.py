"""LORE-KV attention patches, scoring, selection and generation loop.

Use ``replace_*_lookahead`` in :mod:`.monkeypatch` to install the method on a
backbone. Use :func:`.lookahead_runner.generate_with_lookahead` to run
Algorithm 1 on one prompt.
"""
