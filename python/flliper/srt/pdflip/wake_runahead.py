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
from typing import Callable, List, Sequence, Tuple

LINE = "PDFLIP-WAKE-RUNAHEAD"


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


def bound_wait(futs: List[Tuple[str, _cf.Future]], bound: int, any_mode: bool,
               clock: Callable[[], float] = time.perf_counter) -> float:
    """At most ``bound + 1`` collects may stay submitted. FIFO (off): wait
    for the one submitted ``bound + 1`` places back. ANY (on): wait until at
    most ``bound`` are unfinished. Returns the seconds waited."""
    bound = max(1, int(bound))
    if len(futs) <= bound:
        return 0.0
    t0 = clock()
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
