"""P-MINIFWD (30.09.2026): a prompt's small LAST piece waits, bounded, for the
told of a request already in P's queue, so both run in ONE forward.

THE SPECIMEN. NF y3r (c1c012dd5e, ...dauer09292330), PP0 23:53:20-27: six
16450-token requests (weg2-62-92..97) arrive while P sleeps. After the wake
92 runs [0, 16384) (2.9 s); its store probe answers for 93..96 land DURING that
forward (``#1035 R13 EMPTY KV PREFIX`` at 23:53:26, ``#1028 HICACHE-ROUND
ongoing_prefetch=5`` at the top of the next pass). That next pass is the tail
pass of 92: the #1400 told of 93 is not on the wire yet (``pp0_publish`` found
the read still open), so 92's 66-token rest runs ALONE -- 1144.7 gpu-ms on
PP0, of which ~0.8 s is the expert socket of ~4200 spilled experts -- and 93's
chunk 0 follows in the pass after (``confirmed source=told`` at 23:53:27). The
first chunk after a lone rest also fetches its experts throttled (290-357 us
instead of 193 us per expert, +1.0-1.25 s: the PP0 16k outlier class). Every
LATER rest of the same burst already ran inside the next request's chunk 0
(``#969N ADMIT bs=2 extend=16322``) and was not throttled. Per boot: y3r 3,
y3t 4, y3p ~10 such first rests.

THE LEVER. At the top of the tail pass, BEFORE PP0 puts this pass's tolds on
the wire (``weg2_store_told.pp0_publish``), PP0 waits for the first held
request whose store read is still open -- only when

* the carried chunked request's next piece is its FINAL one (no END-ANCHOR
  split, which would truncate the piece and bar every co-admission) and
  smaller than :data:`MINI_TOKENS` (the class the order names: P forwards
  below 1000 tokens at bs 1);
* no queued request is admissible already (it would join without waiting);
* a queued request is held with an OPEN store read (no twin deferral, no
  A12.2 deferral): "without a waiter, compute at once";
* the pass carries no flip/flush/abort control request (the flip is never
  slowed).

THE BOUND IS MEASURED, never a constant: the wait may cost at most what the
lone forward it replaces costs -- the running mean of this rank's own
forwards below :data:`MINI_TOKENS` (``RankPrefillLog`` gpu-ms, fed by
:func:`note_forward`). Ski rental: waiting up to the price of the alternative
is never worse than twice the optimum. Unmeasured (no mini forward seen on
this boot yet) -> no wait. Every hold prints what it waited and how it ended.

RANK-UNIFORM BY CONSTRUCTION. Only PP0 waits, and only BEFORE its tolds go on
the wire; the followers admit on the told objects of the pass list exactly as
without the wait (the told arrival IS the membership signal, #1400), so no
follower ever sees a rank-local clock.

Narrow arguments (large-class-style 1.6): :func:`decide` is pure; the
scheduler-facing adapter lives in ``weg2_store_told``.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: The order's class: a P forward below 1000 tokens (bs 1) is dominated by
#: the expert socket (~0.8 s on PP0) -- the rests this module folds.
MINI_TOKENS = 1000
#: How often the wait looks at the open read (host poll; the read itself runs
#: on the prefetch thread).
POLL_S = 0.002
#: Running mean of the measured lone-rest cost: new = old + (x - old) / min(n, W).
_MEAN_WINDOW = 16
#: Control requests that must never wait behind a hold (flip, flush, abort).
CONTROL_KINDS = (
    "FlushCacheReqInput",
    "ReleaseMemoryOccupationReqInput",
    "ResumeMemoryOccupationReqInput",
    "AbortReq",
)

_STATE = {"mini_ms": None, "n_mini": 0, "n": 0, "told": 0, "timeout": 0, "waited_ms": 0.0}


def enabled() -> bool:
    return bool(envs.SGLANG_WEG2_P_MINIFWD_TOLD_WAIT.get())


def note_forward(new_tokens: int, chunks: int, gpu_ms: float) -> None:
    """``RankPrefillLog``: one prefill line of this rank. A single forward
    below :data:`MINI_TOKENS` is a lone-rest price sample."""
    try:
        new_tokens, chunks, gpu_ms = int(new_tokens), int(chunks), float(gpu_ms)
    except (TypeError, ValueError):
        return
    if chunks != 1 or not (0 < new_tokens < MINI_TOKENS) or gpu_ms <= 0.0:
        return
    n = _STATE["n_mini"] + 1
    _STATE["n_mini"] = n
    old = _STATE["mini_ms"]
    _STATE["mini_ms"] = gpu_ms if old is None else old + (gpu_ms - old) / min(n, _MEAN_WINDOW)


def wait_bound_s() -> Optional[float]:
    """The measured price of a lone rest in seconds, or None (unmeasured)."""
    ms = _STATE["mini_ms"]
    return None if ms is None else float(ms) / 1000.0


def reset() -> None:
    """Tests: forget the measurements and the counters."""
    _STATE.update({"mini_ms": None, "n_mini": 0, "n": 0, "told": 0, "timeout": 0, "waited_ms": 0.0})


class Verdict(NamedTuple):
    wait: bool
    reason: str
    rest: int
    rids: tuple


def final_rest(fill_len: int, done: int, chunk_budget: Optional[int], split: bool) -> Optional[int]:
    """Tokens of the carried request's next piece when it is its FINAL piece,
    else None: the rest must fit the chunk budget and must not be split by
    the END-ANCHOR (a split piece is truncated and admits nobody beside it)."""
    rest = int(fill_len) - int(done)
    if rest <= 0 or split:
        return None
    if chunk_budget is not None and int(chunk_budget) > 0 and rest > int(chunk_budget):
        return None
    return rest


def decide(
    *,
    rest: Optional[int],
    queued: Sequence[str],
    admissible: Callable[[str], bool],
    open_read: Callable[[str], bool],
    control: bool,
    bound_s: Optional[float],
) -> Verdict:
    """Hold this pass for a told? ``queued`` in queue order; ``admissible``
    = its told is stored (it joins without waiting); ``open_read`` = held by
    #1400 with a store read still running (no deferral)."""
    if rest is None:
        return Verdict(False, "no-final-rest", 0, ())
    if rest >= MINI_TOKENS:
        return Verdict(False, "rest-not-mini", rest, ())
    if control:
        return Verdict(False, "control", rest, ())
    if any(admissible(r) for r in queued):
        return Verdict(False, "joins-anyway", rest, ())
    waiters = tuple(r for r in queued if open_read(r))
    if not waiters:
        return Verdict(False, "no-waiter", rest, ())
    if bound_s is None or bound_s <= 0.0:
        return Verdict(False, "unmeasured", rest, waiters)
    return Verdict(True, "hold", rest, waiters)


def wait(
    waiters: Iterable[str],
    terminable: Callable[[str], bool],
    bound_s: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple:
    """Poll until one waiter's read can terminate or the bound runs out.
    Returns ``(outcome, rid or None, waited_s)``; outcome 'told' | 'timeout'."""
    ws: List[str] = list(waiters)
    t0 = clock()
    while True:
        for rid in ws:
            try:
                if terminable(rid):
                    return "told", rid, clock() - t0
            except Exception as exc:  # noqa: BLE001 -- a broken probe ends the hold, never the pass
                logger.warning("P-MINIFWD-HOLD probe failed rid=%s (%s: %s)", rid, type(exc).__name__, exc)
                return "timeout", None, clock() - t0
        waited = clock() - t0
        if waited >= bound_s:
            return "timeout", None, waited
        sleep(min(POLL_S, max(0.0, bound_s - waited)))


def log_hold(owner_rid: str, verdict: Verdict, outcome: str, rid: Optional[str], waited_s: float,
             bound_s: float) -> None:
    n = _STATE["n"] + 1
    _STATE["n"] = n
    _STATE[outcome] = _STATE.get(outcome, 0) + 1
    _STATE["waited_ms"] += waited_s * 1000.0
    if n <= 16 or n % 64 == 0 or outcome != "told":
        logger.info(
            "WEG2 P-MINIFWD-HOLD n=%d rid=%s rest=%d outcome=%s waiter=%s waited_ms=%.1f "
            "bound_ms=%.1f (measured lone rest, n_mini=%d) waiters=%d told=%d timeout=%d "
            "waited_ms_sum=%.0f: the rest runs in the waiter's chunk 0 when its told lands "
            "in this pass",
            n, str(owner_rid), verdict.rest, outcome, rid, waited_s * 1000.0,
            bound_s * 1000.0, _STATE["n_mini"], len(verdict.rids), _STATE["told"],
            _STATE["timeout"], _STATE["waited_ms"],
        )
