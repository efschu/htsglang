"""X-SUM-PRICE: X bounds what D prefills IN TOTAL -- the one rule.

User 05.10.: "pending token werden gesammelt und als SUMME bepreist". The rule
grew in five places, each with its own comparison (the queue take, DECODE-COLLECT's
release and hand-over, D-SHORT-DRAIN, the idle re-grant). Every one of them now
asks ``fits``; the queue take is ``take_oldest_first``. ``carried`` is what D
already has to prefill (granted SHORTs without a first token, requests handed
to D and not started): see ``Front._d_carried_tokens``.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def fits(*, carried: int, tokens: int, limit: int) -> bool:
    """May D prefill ``tokens`` more while it still has ``carried`` to do?"""
    return int(carried) + int(tokens) <= int(limit)


def take_oldest_first(entries: Sequence[T], *, tokens_of: Callable[[T], int],
                      arrived_of: Callable[[T], float], carried: int,
                      limit: int) -> Tuple[List[T], List[T]]:
    """Walk ``entries`` oldest first, take each one that still fits on top of
    ``carried`` and the ones taken before it; the rest is kept back (it stays
    queued, the flip to P takes it)."""
    taken: List[T] = []
    kept: List[T] = []
    total = int(carried)
    for e in sorted(entries, key=arrived_of):
        u = int(tokens_of(e) or 0)
        if fits(carried=total, tokens=u, limit=limit):
            taken.append(e)
            total += u
        else:
            kept.append(e)
    return taken, kept
