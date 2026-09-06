"""Make a modern tokenizer.json readable by the pinned tokenizers 0.19.

transformers 4.44 pins `tokenizers>=0.19,<0.20`. Files written by tokenizers
>= 0.20 store BPE merges as ["a", "b"] pairs; 0.19 only parses the older "a b"
string form and fails the whole load with

    Exception: data did not match any variant of untagged enum ModelWrapper

Both Qwen3 and Ministral-3 ship the new form. The two encodings are equivalent
for byte-level BPE, where no merge piece can contain a literal space (spaces are
the 'G-dot' byte), so the join back to a string is unambiguous -- and
`downgrade_merges` refuses rather than guesses if that ever stops being true.

This is a file-format downgrade only. It does not touch the vocabulary, the
pre-tokenizer regex, the added tokens, or the decoder, so the resulting tokenizer
produces identical ids; scripts/validate_models_ext.py checks that against the
original file read by a modern tokenizers.
"""


def downgrade_merges(tok_json):
    """["a","b"] merge pairs -> "a b" strings, in place. Returns the number rewritten."""
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
