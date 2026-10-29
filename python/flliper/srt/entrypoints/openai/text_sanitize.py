"""Lone UTF-16 surrogates in client text and in parsed tool-call arguments.

A JSON string may carry ``"\\ud83d"`` without its low half (an agent client
that truncates a JS string in the middle of an emoji emits exactly that, and a
JSON-format tool-call model can write a broken pair into its arguments).
``json.loads`` accepts it and yields a ``str`` that cannot be encoded as
UTF-8: the HF tokenizer rejects it with ``TypeError: TextInputSequence must
be str`` and the response encoders (orjson, Starlette's JSONResponse) raise,
so the turn answers a bare 500 -- and, because the broken text stays in the
client's history, every later turn too. Each lone surrogate is replaced by
U+FFFD, which is what an incremental detokenizer shows for a cut-off emoji.
Text without a lone surrogate is returned unchanged (same object).
"""

import re
from typing import Any

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
REPLACEMENT_CHARACTER = "�"


def scrub_lone_surrogates(value: Any) -> Any:
    """``value`` with every lone surrogate in its strings replaced.

    Walks ``str``, ``list`` and ``dict`` (keys and values); anything else is
    returned as is. A container is rebuilt only when something inside it
    changed, so the common case allocates nothing.
    """
    if isinstance(value, str):
        if value.isascii():
            return value
        return _LONE_SURROGATE.sub(REPLACEMENT_CHARACTER, value)
    if isinstance(value, list):
        items = [scrub_lone_surrogates(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return items
    if isinstance(value, dict):
        entries = {
            scrub_lone_surrogates(key): scrub_lone_surrogates(item)
            for key, item in value.items()
        }
        if all(
            new_key is old_key and new_item is old_item
            for (new_key, new_item), (old_key, old_item) in zip(
                entries.items(), value.items()
            )
        ):
            return value
        return entries
    return value
