"""xsn328/329 (18.09.2026): D's dormant re-reads of a held request answered
zero although P had completed the pages -- D's own page keys (hashed from
its ids) matched P's for the first 64 tokens only. P hands its page keys
over (#1442 handoff file); the dormant prefetch uses THOSE keys, so the
store is asked for exactly the pages P wrote. Pure helpers, desk-testable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def keys_for_span(page_keys: Optional[Sequence[str]], matched_len: int, n_tokens: int,
                  page_size: int = 1) -> Optional[List[str]]:
    """The handed-over keys of the pages covering tokens
    [matched_len, matched_len + n_tokens), or None when the hand-off does not
    cover that span (then the caller hashes as before)."""
    if not page_keys or n_tokens <= 0 or page_size <= 0:
        return None
    if matched_len % page_size:
        return None
    p0 = matched_len // page_size
    n_pages = (n_tokens + page_size - 1) // page_size
    if p0 + n_pages > len(page_keys):
        return None
    return [str(k) for k in page_keys[p0:p0 + n_pages]]


def first_mismatch(a: Sequence[str], b: Sequence[str]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
