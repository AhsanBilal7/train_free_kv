"""Convert a tokenizer.json to the format that tokenizers 0.19 can read.

transformers 4.44 requires ``tokenizers>=0.19,<0.20``. tokenizers 0.20 and later
store each BPE merge as a pair ["a", "b"]. Release 0.19 reads only the string
format "a b" and fails with::

    Exception: data did not match any variant of untagged enum ModelWrapper

The Ministral-3 tokenizer uses the pair format. For byte-level BPE no merge piece
contains a space, so the conversion is exact. :func:`downgrade_merges` raises an
error if a piece contains a space.

The conversion changes only the file format. The vocabulary, pre-tokenizer, added
tokens and decoder do not change, so the token IDs are the same.
``scripts/validate_models_ext.py`` checks this.
"""


def downgrade_merges(tok_json):
    """Convert ["a", "b"] merge pairs to "a b" strings in place.

    Returns:
        The number of converted merges.
    """
    model = tok_json.get("model", {})
    merges = model.get("merges")
    if not merges or isinstance(merges[0], str):
        return 0
    out = []
    for pair in merges:
        if len(pair) != 2:
            raise ValueError(f"unexpected merge entry {pair!r}: expected exactly two pieces")
        left, right = pair
        if " " in left or " " in right:
            raise ValueError(f"merge piece contains a literal space: {pair!r}; cannot join losslessly")
        out.append(f"{left} {right}")
    model["merges"] = out
    return len(out)
