"""SEQ-HASH (D-NORECOMPUTE (d), 02.10.): a follow-up turn that EXTENDS the
previous turn's GENERATED tokens is credited with D's realised anchor on that
sequence, not only with the previous PROMPT.

y6y: pdflip-18-117 (prompt 85001, 4379 decoded) finished with
'PRESENCE-OWN-TEXT-CLAMP ... prompt_d=85001 depth_d=89344 credited=84992' --
the front had only the prompt ids, so it clamped D's depth to the prompt.
pdflip-18-130 (89398 tokens, 2 s later) priced uncached 4406 > X=4305 (LONG), and
P read exactly 89344 from the store and computed 54. pdflip-16-92 (79960) the same
shape on pdflip-12-44's decode path (park depths 78080 / 79104, finish 79872).

D's tokenizer manager (never the scheduler, so decode is untouched) hashes the
request's whole sequence -- prompt ids + output ids -- up to the depth D names
as resumable: at the finish (``meta_info.pdflip_seq_hash``, beside
``pdflip_resumable_depth``) and at a park (``pdflip_seq_hash`` {rid: mark} beside
the #59b depths). A mark is ``"<depth>:<hash>"``. The front hashes a new
prompt's ids to the same depth with the same function: equal = the prompt
extends that sequence to there, and D's anchor at that depth is credited.

The function is sha256 over the little-endian int32 token ids (0.2 ms for 90k
ids measured) -- an equality witness of two id sequences, not a store key.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

FIELD = "pdflip_seq_hash"
_HEX = 32


def digest(ids: Sequence[int], depth: int) -> Optional[str]:
    """The hash of ``ids[:depth]`` (None when ``ids`` is shorter or depth <= 0)."""
    depth = int(depth)
    if depth <= 0:
        return None
    arr = ids if isinstance(ids, np.ndarray) else np.asarray(list(ids), dtype=np.int32)
    if arr.size < depth:
        return None
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32)
    return hashlib.sha256(np.ascontiguousarray(arr[:depth]).astype("<i4").tobytes()).hexdigest()[:_HEX]


def mark(ids: Sequence[int], depth: Optional[int]) -> Optional[str]:
    """``"<depth>:<hash>"`` for the sequence up to ``depth``, or None."""
    if depth is None:
        return None
    h = digest(ids, int(depth))
    return f"{int(depth)}:{h}" if h else None


def parse(value) -> Optional[Tuple[int, str]]:
    """``(depth, hash)`` of a mark, None when absent or malformed."""
    if not isinstance(value, str) or ":" not in value:
        return None
    d, h = value.split(":", 1)
    try:
        depth = int(d)
    except ValueError:
        return None
    if depth <= 0 or len(h) != _HEX:
        return None
    return depth, h


def sequence(prompt_ids: Optional[Iterable[int]], output_ids: Optional[Iterable[int]]) -> Optional[list]:
    """The tokenizer manager's view of a request's whole sequence."""
    if prompt_ids is None:
        return None
    return list(prompt_ids) + list(output_ids or ())
