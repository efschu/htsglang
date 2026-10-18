"""The uncached tokens D has been granted to prefill and has not finished yet.

X-SUM-PRICE (user 05.10.): X bounds what D prefills IN TOTAL, not per request.
The queue paths already price their sum (``_sum_priced_take``); the ARRIVAL
path did not: a SHORT granted to an idle D was invisible to the next arrival
until its first token, because DECODE-COLLECT opens only while D *decodes*.
Six 6239-token turns arriving together were 37k tokens of D prefill, each one
under X=12288 (NF y6 acceptance 09.10., ``PDFLIP-ROUTE ... SHORT -> D`` x6 in one
second, no ``DECODE-COLLECT`` line).

This book is that missing term: grant at the SHORT verdict, ``enter_leg2`` when
D has the request, ``done`` at its first content (D's prefill is over). The
front reads ``pending_tokens`` as ``carried`` for DECODE-COLLECT.
"""
from __future__ import annotations

from typing import Callable, Container, Dict, Optional

import msgspec

#: A grant that has not reached leg 2 (seat wait, price barrier, early flip)
#: counts this long, then it is dropped: a request that left before leg 2
#: (client gone, fell through to BATCH) must not hold D's sum up for ever.
GRANT_TTL_S = 30.0


class _Row(msgspec.Struct):
    tokens: int
    t_grant: float
    in_leg2: bool = False


class DPrefillInflight:
    def __init__(self) -> None:
        self._rows: Dict[str, _Row] = {}

    def grant(self, *, rid: str, tokens: int, now: float) -> None:
        self._rows[rid] = _Row(tokens=max(0, int(tokens)), t_grant=now)

    def enter_leg2(self, *, rid: str) -> None:
        row = self._rows.get(rid)
        if row is not None:
            row.in_leg2 = True

    def done(self, *, rid: str) -> None:
        self._rows.pop(rid, None)

    def is_open(self, *, rid: str) -> bool:
        """D's prefill of ``rid`` is not over: granted and no first content yet (NODECODE 1015: such a rid in D's
        ledger is ``carried``, not a decode -- ``Front._dc_decoding``)."""
        return rid in self._rows

    def pending_tokens(self, *, now: float, live: Container[str],
                       idle: Optional[Callable[[str], bool]] = None,
                       started: Optional[Callable[[str, float], bool]] = None) -> int:
        """Sum of the open rows. A row in leg 2 lives while D still holds the
        rid (``live`` = D's outstanding set); one that never got there lives
        ``GRANT_TTL_S``. Dead rows are dropped here. ``idle(rid)`` True = the
        row lives but D is not prefilling it now (parked, held below a raised
        lane floor): it is kept and not counted. ``started(rid, t_grant)``
        (S49, SOLL R6 "nur echt laufend zählt"): given, a living row counts
        only when D really began its extend (granted-but-not-started is kept,
        not counted)."""
        total = 0
        for rid in list(self._rows):
            row = self._rows[rid]
            alive = rid in live if row.in_leg2 else now - row.t_grant < GRANT_TTL_S
            if alive:
                if idle is not None and idle(rid):
                    continue
                if started is not None and not started(rid, row.t_grant):
                    continue
                total += row.tokens
            else:
                del self._rows[rid]
        return total
