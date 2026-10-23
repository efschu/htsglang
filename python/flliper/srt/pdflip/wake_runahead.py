"""WAKE-RUNAHEAD-ANY (30.09., NF y4k/y4l P->D): the wake loop's run-ahead
bound waits for ANY collect in flight, not for the OLDEST one.

THE MEASUREMENT (y4l n=12 / y4k n=14 P->D flips). The waking group's main
thread resumes tag t and submits its collect; the bound (weg2xsn110) keeps at
most ``bound + 1`` collects submitted and, at the limit, waited for the
collect submitted ``bound + 1`` places earlier -- whatever else finished. Under
the source round-robin order (587a0de31e) that oldest collect is often a band
of a SLOW source (a 3080 P stage: PP2 weights_14 deposit 0.268-0.512 s, PP1
weights_12 0.557-0.751 s in y4l epoch 4) while the newer ones -- among them
PP0's bands, the flip's critical chain (8.4 GB through the 5090's one D2H
engine) -- are done. D TP1 then resumed PP0's weights_3 at 0.725 s although
its collect of PP0's weights_2 ended ~0.65 s; PP0's p0 deposit into D TP1
waited for the collector's slots from 0.651 s (credit). Per flip: the D
3080 ranks' main loop stood 654 / 659 ms (y4l median; credit waits 0 / 122)
between resumes, and PP0's two BAR1 deposit lanes waited 313 / 138 ms for
credits (y4k 320 / 143) -- in the leg whose end is PP0's chain end.

THE FORM (``FLLIPER_PDFLIP_WAKE_RUNAHEAD_ANY``, per group env, default off =
the FIFO wait byte for byte): at the bound, wait until the number of
UNFINISHED collects is at most ``bound`` (``FIRST_COMPLETED``), raising the
first failed collect it sees. The same number of collects in flight, the
same resumes in the same order, every resume still behind its VRAM credit
wait: no new VRAM, no reserve. Only WHEN the loop continues changes -- never
later than the FIFO wait (the oldest collect finishing is one of the events
it waits for).
"""

from __future__ import annotations

import concurrent.futures as _cf
import time
from typing import Callable, List, Optional, Sequence, Tuple

LINE = "PDFLIP-WAKE-RUNAHEAD"

# -- WAKE-RUNAHEAD-DEPOSITOR (barlink-lanes2-1023) ------------------------------
#
# THE MEASUREMENT (boot dkr27bnvfp4duorebarbar1duo10230932, ACS 0011, 300 BAR1
# tag pairs, 49 under 10 GB/s). In the start-offset class the depositor fills
# its ring (4 slots) in a few ms and then waits for ``free(0)``: its collector
# has not begun the tag. A collector begins tag t only after the wake loop's
# main thread has resumed t, and that thread sits in the run-ahead bound above
# (FIFO on the collect ``bound + 1`` places back) while the bytes wait on the
# wire. Worst tag, P PP0 lane p1 ``8-weights_4``: depositor D TP1 began at
# 1792748993.904, PP0's main thread waited 354 ms in the bound for weights_1's
# collect (held by its host/IPC lane c0, not by p1), resumed weights_4 at
# .162-.172, the collect began .177 -- the depositor's credit wait was 271 of
# its 295 ms (2.1 GB/s for a lane that moves 25-30 GB/s).
#
# THE FORM (``FLLIPER_PDFLIP_WAKE_RUNAHEAD_DEPOSITOR``, default on): the bound
# wait ALSO ends when the NEXT tag's BAR1 depositor already waits for its
# first free (``early()``, the depositor's ``ffwait`` flag), but only while at
# most ``bound + 1`` collects are unfinished -- so at most ONE collect beyond
# the bound is ever in flight, and only for a tag whose bytes are already
# waiting. Every resume still goes through its VRAM credit wait first; the
# wake loop adds one collect worker so the early tag does not queue behind the
# others. Off = the bound byte for byte.
#: The bound's poll slice while ``early`` is armed: one stat per receiving lane
#: per slice. 5 ms = the interpreter's switch interval, so the waiting main
#: thread asks for the GIL no more often than a busy thread would be preempted
#: anyway (the collectors' issue spikes were GIL competition); the reaction
#: costs <= 5 ms against offsets of 60-354 ms.
EARLY_POLL_S = 0.005


def runahead_depositor_on() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_WAKE_RUNAHEAD_DEPOSITOR.get())
    except Exception:  # noqa: BLE001 -- no environ: the bound as before
        return False


def runahead_any_on() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_WAKE_RUNAHEAD_ANY.get())
    except Exception:  # noqa: BLE001 -- no environ: the FIFO wait
        return False


def _raise_failed(futs: Sequence) -> None:
    for f in futs:
        if f.done() and not f.cancelled() and f.exception() is not None:
            f.result()  # re-raise the collect's own exception


def _unfinished(futs: Sequence) -> int:
    return sum(1 for _t, f in futs if not f.done())


def bound_wait(futs: List[Tuple[str, _cf.Future]], bound: int, any_mode: bool,
               clock: Callable[[], float] = time.perf_counter,
               early: Optional[Callable[[], bool]] = None,
               stats: Optional[dict] = None) -> float:
    """At most ``bound + 1`` collects may stay submitted. FIFO (off): wait
    for the one submitted ``bound + 1`` places back. ANY (on): wait until at
    most ``bound`` are unfinished. Returns the seconds waited.

    ``early`` (WAKE-RUNAHEAD-DEPOSITOR): asked while the wait lasts; truthy
    while at most ``bound + 1`` collects are unfinished ends the wait (the
    next tag's depositor already waits for this rank); a failed collect is
    raised as soon as it is seen, as in ANY. ``stats["early"]`` counts those
    releases."""
    bound = max(1, int(bound))
    if len(futs) <= bound:
        return 0.0
    t0 = clock()
    if early is not None:
        return _bound_wait_early(futs, bound, any_mode, clock, early, stats, t0)
    if not any_mode:
        futs[-(bound + 1)][1].result()
        return clock() - t0
    all_f = [f for _t, f in futs]
    _raise_failed(all_f)
    pending = [f for f in all_f if not f.done()]
    while len(pending) > bound:
        done, _ = _cf.wait(pending, return_when=_cf.FIRST_COMPLETED)
        _raise_failed(done)
        pending = [f for f in pending if not f.done()]
    return clock() - t0


def _bound_wait_early(futs, bound, any_mode, clock, early, stats, t0) -> float:
    """The bound with the depositor release: the same condition as
    :func:`bound_wait` ends it, or ``early()`` while <= ``bound + 1`` are
    unfinished. Polls in EARLY_POLL_S slices on the futures (a done future
    returns at once); ``early`` is a stat of one flag file per BAR1 lane."""
    target = futs[-(bound + 1)][1]
    all_f = [f for _t, f in futs]
    while True:
        _raise_failed(all_f)
        if any_mode:
            if _unfinished(futs) <= bound:
                return clock() - t0
        elif target.done():
            target.result()
            return clock() - t0
        if _unfinished(futs) <= bound + 1:
            try:
                go = bool(early())
            except Exception:  # noqa: BLE001 -- a probe never fails the wake
                go = False
            if go:
                if stats is not None:
                    stats["early"] = int(stats.get("early", 0)) + 1
                return clock() - t0
        pending = [f for f in all_f if not f.done()]
        if pending:
            _cf.wait(pending, timeout=EARLY_POLL_S, return_when=_cf.FIRST_COMPLETED)
