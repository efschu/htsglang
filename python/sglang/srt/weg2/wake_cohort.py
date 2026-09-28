# SPDX-License-Identifier: Apache-2.0
"""L1 (28.09.): the post-wake cohort -- the reads of one wake join ONE extend.

NF rc12z26 D (boot ...09281752, TP0, 18:06:01): the wake issued six store
reads (``#248 WAKE-READ issued=6``), all parked by the ``#1471`` settle. The
reads took 0.1-0.7 s. weg2-3-8 finished first (430 ms), was released ALONE and
ran its extend alone: 29 new tokens, ``gpu-ms 2002``. The five others finished
inside that pass and waited it out (``WEG2-LOAD-DEVICE harvest_ms`` 2054-2167),
then ran two more extends of 1552 / 1612 ms. The last request decoded ~6 s
after the wake although every byte was on the host after 0.7 s.

THE PRICE IS PER PASS, NOT PER TOKEN. Every D extend with >= 8 new tokens costs
1.5-2.2 s (rc12z25: n=261, median 2188 ms, p10 1552; rc12z26: n=67, median
1801, p10 1492) -- the eager expert-major prefill streams the routed spill
experts of every layer over PCIe (``MoE offload layer 0: expert-major prefill,
3 waves, 0.20 GiB H2D``), and 29 tokens already route to most of them. An
extend under 8 new tokens (the E2 tail skip, a 1-4 token E1) costs ~12 ms.
So a member released alone into an expensive pass makes every sibling that
lands during it wait the whole pass; held back until its siblings land, the
cohort runs that price once.

THE RULE (27B operator, 28.09.): hold a READY member of a wake's settle only
  * while a SIBLING read of the same settle is still in flight
    (``reading`` / ``reissued``; ``wait`` = a writer at work has no ETA and
    never holds the cohort),
  * when a ready member's own pass would be the expensive one (>= 8 new
    tokens after its read; an E2 skip / small E1 is released at once),
  * and never longer than :data:`COHORT_S` (1.0 s) nor than HALF the pass it
    saves: the price is MEASURED on this rank (:func:`note_extend`, an EMA of
    the extends with >= 8 new tokens, fed by the ``Prefill rank batch`` gpu-ms
    instrument) -- NF's LRU-aware eager plan will bring it from ~1.8 s to
    ~0.9 s, and the hold shrinks with it. :data:`PRICE_S` (the rc12z26 p10,
    1.5 s) stands in only until the first extend was measured.
A single parked request (a wake of one) never waits -- the vote is not even
asked.

GROUP-UNIFORM (#580/#791): whether the vote is asked depends only on the
replicated settle list's length and the switch; the vote itself is one more
element in the settle tick's EXISTING ``_weg2_group_min_flags`` all_reduce
(no new collective), and the group holds only when EVERY rank votes hold --
any rank whose clock lapsed, or that sees no sibling in flight, releases the
group. Rank-local clocks feed the vote, never the decision.

Switch: ``SGLANG_WEG2_WAKE_COHORT`` (default on; 0 = the release of each
member the moment its read completes, as before).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_WAKE_COHORT"
COHORT_S_ENV = "SGLANG_WEG2_WAKE_COHORT_S"
PRICE_S_ENV = "SGLANG_WEG2_WAKE_COHORT_PRICE_S"
HOLD_FRACTION_ENV = "SGLANG_WEG2_WAKE_COHORT_FRACTION"
#: hard cap of one hold (27B operator: "COHORT_S 1,0 s als harte Kappe").
COHORT_S = 1.0
#: the pass the hold saves before one was measured: p10 of NF D's extends with >= 8 new
#: tokens (rc12z26 1492 ms, rc12z25 1552 ms).
PRICE_S = 1.5
#: the hold is at most this share of the measured pass price -- strictly below the pass it saves.
HOLD_FRACTION = 0.5
#: EMA weight of a new extend measurement.
PRICE_ALPHA = 0.3
_PRICE = {"ms": None, "n": 0}
#: below this many new tokens the member's own pass is the cheap one (~12 ms: E2 / small E1).
EXPENSIVE_MIN_TOKENS = 8
#: a sibling in one of these states has a read in flight that will land by itself.
IN_FLIGHT = ("reading", "reissued")


def enabled(env=None) -> bool:
    raw = ((env if env is not None else os.environ).get(ENV, "1") or "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _env_float(name: str, default: float, env=None) -> float:
    try:
        return float((env if env is not None else os.environ).get(name, "") or default)
    except ValueError:
        return default


def note_extend(new_tokens: int, gpu_ms: float) -> None:
    """The metrics reporter's ``Prefill rank batch`` line: an extend of the
    expensive class feeds the price EMA (a cheap one says nothing about it)."""
    if int(new_tokens) < EXPENSIVE_MIN_TOKENS or not gpu_ms or gpu_ms <= 0:
        return
    prev = _PRICE["ms"]
    _PRICE["ms"] = float(gpu_ms) if prev is None else (1.0 - PRICE_ALPHA) * prev + PRICE_ALPHA * float(gpu_ms)
    _PRICE["n"] += 1


def price_s(env=None) -> float:
    """The pass a hold saves, in seconds: measured when an extend was, else PRICE_S."""
    if _PRICE["ms"] is not None:
        return _PRICE["ms"] / 1000.0
    return _env_float(PRICE_S_ENV, PRICE_S, env)


def cap_s(env=None) -> float:
    """The longest a hold may run: the hard cap, and at most HOLD_FRACTION of the pass it saves."""
    frac = min(1.0, max(0.0, _env_float(HOLD_FRACTION_ENV, HOLD_FRACTION, env)))
    return max(0.0, min(_env_float(COHORT_S_ENV, COHORT_S, env), frac * price_s(env)))


def asks(settle: Sequence, env=None) -> bool:
    """Whether the tick puts the cohort vote on the wire. Rank-uniform by
    construction: the settle list is replicated in arrival order and the
    switch is the launcher's env -- so every rank sends the same number of
    flags into the one MIN."""
    return enabled(env) and len(settle) >= 2


def new_tokens(req, records=None) -> Optional[int]:
    """Tokens ``req``'s own extend computes once its read is in: the prompt
    minus the materialized prefix (``PrefetchOutcome.materialized``, the
    group-synced record), or 0 when the group agreed the E2 tail skip. None =
    unknown (no annotated record) -- the caller treats it as cheap, so an
    unknown never buys a wait."""
    ids = getattr(req, "full_untruncated_fill_ids", None)
    if ids is None:
        return None
    rec = None if records is None else records.get(getattr(req, "rid", None))
    mat = getattr(rec, "materialized", None)
    if mat is None:
        return None
    try:
        from sglang.srt.weg2 import tail_adopt as _ta

        start, _wait = _ta.peek_target_start(req, int(mat), batch_empty=True)
    except Exception:  # noqa: BLE001 -- no adopt view = the page anchor, as the admission would
        start = None
    first = int(mat) if start is None else int(start)
    return max(0, len(ids) - first)


def hold_vote(members: Iterable[Tuple[object, str, bool]], since: float, now: float,
              records=None, env=None) -> int:
    """This rank's vote (1 = hold the ready members this tick). ``members`` =
    (req, read state, this rank's release flag) for every parked request;
    ``since`` = the earliest park time of the settle (the wake)."""
    ready, flying = [], 0
    for req, state, release in members:
        if release:
            ready.append(req)
        elif state in IN_FLIGHT:
            flying += 1
    if not ready or not flying:
        return 0
    if now - since >= cap_s(env):
        return 0
    for req in ready:
        n = new_tokens(req, records)
        if n is not None and n >= EXPENSIVE_MIN_TOKENS:
            return 1
    return 0


class WakeLedger:
    """One line per wake: wake -> the last cohort member's first decode, with
    the cohort's hold, so the metal shows before/after in one number."""

    def __init__(self) -> None:
        self.seq: Optional[int] = None
        self.t_wake = 0.0
        self.rids: List[str] = []
        self.hold_t0: Optional[float] = None
        self.hold_ms = 0.0
        self.holds = 0
        self.open = False

    def arm(self, seq, rids: Sequence[str], t_wake: float) -> None:
        self.seq, self.t_wake, self.rids = seq, float(t_wake), [str(r) for r in rids]
        self.hold_t0, self.hold_ms, self.holds = None, 0.0, 0
        self.open = bool(self.rids)

    def note_hold(self, now: float, held: bool) -> None:
        if held:
            if self.hold_t0 is None:
                self.hold_t0 = now
                self.holds += 1
        elif self.hold_t0 is not None:
            self.hold_ms += (now - self.hold_t0) * 1000.0
            self.hold_t0 = None

    def decode_seen(self, batch_rids: Iterable[str], now: float, cohort_on: bool) -> Optional[str]:
        """The line, once, when every cohort member is in a decode batch."""
        if not self.open:
            return None
        have = {str(r) for r in batch_rids}
        if any(r not in have for r in self.rids):
            return None
        self.open = False
        self.note_hold(now, False)
        return ("WEG2-WAKE-COHORT wake=%s n=%d wake_to_last_decode_ms=%.0f hold_ms=%.0f holds=%d "
                "cohort=%s rids=%s (L1: the reads of one wake join one extend; before L1 the "
                "first finished read ran its 1.5-2.2 s expert pass alone)"
                % (self.seq, len(self.rids), (now - self.t_wake) * 1000.0, self.hold_ms, self.holds,
                   "on" if cohort_on else "off", [r[:12] for r in self.rids[:8]]))


def monotonic() -> float:
    return time.monotonic()
