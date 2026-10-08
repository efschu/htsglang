"""WAKE-RUNAHEAD-ANY (30.09., NF P->D): the wake loop's run-ahead bound waits
for ANY collect in flight, not for the oldest one.

Metal (y4l 11e5db4370 n=12 / y4k dff1a7fed4 n=14 P->D flips): D TP1/TP2's
main loop stood 654 / 659 ms per flip between resumes (credit waits 0 / 122),
waiting for the OLDEST collect -- often a slow 3080 source's band (y4l epoch
4: PP2 weights_14 0.268-0.512 s) -- while newer collects, PP0's among them,
were done; PP0's BAR1 deposit lanes p0 / p1 waited 313 / 138 ms for the
collectors' credits (y4k 320 / 143), inside PP0's chain, the leg's end.

Pinned: off = the FIFO wait (blocks on the oldest even when newer ones are
done); on = returns as soon as at most ``bound`` collects are unfinished,
never later than FIFO, re-raises a failed collect; the wiring in the wake
loop and the WAKE-OVERLAP line.
"""

import concurrent.futures as cf
import pathlib
import threading
import time

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import wake_runahead as wr

REPO = pathlib.Path(__file__).resolve().parents[4]


def _futs(*states):
    """states: 'done' | 'pending' | Exception; oldest first."""
    out = []
    for i, s in enumerate(states):
        f = cf.Future()
        if s == "done":
            f.set_result(None)
        elif isinstance(s, BaseException):
            f.set_exception(s)
        out.append((f"weights_{i}", f))
    return out


def _later(f, s):
    threading.Timer(s, lambda: f.set_result(None)).start()


def test_switch_default_off():
    assert envs.FLLIPER_PDFLIP_WAKE_RUNAHEAD_ANY.get() is False
    assert wr.runahead_any_on() is False


def test_below_the_bound_nothing_waits():
    futs = _futs("pending", "pending")
    assert wr.bound_wait(futs, 2, False) == 0.0
    assert wr.bound_wait(futs, 2, True) == 0.0


def test_fifo_blocks_on_the_oldest_although_newer_are_done():
    """The defect: the slow source's band holds the loop."""
    futs = _futs("pending", "done", "done")
    _later(futs[0][1], 0.3)
    waited = wr.bound_wait(futs, 2, False)
    assert waited >= 0.25


def test_any_goes_on_when_at_most_bound_are_unfinished():
    futs = _futs("pending", "done", "done")
    _later(futs[0][1], 0.3)
    assert wr.bound_wait(futs, 2, True) < 0.05
    futs[0][1].result(1)


def test_any_waits_for_the_first_completion_not_the_oldest():
    futs = _futs("pending", "pending", "pending")
    _later(futs[0][1], 0.6)   # the slow band
    _later(futs[2][1], 0.1)   # PP0's band, submitted last, done first
    t0 = time.perf_counter()
    waited = wr.bound_wait(futs, 2, True)
    assert 0.05 <= waited < 0.4 and time.perf_counter() - t0 < 0.4
    futs[0][1].result(1)


def test_any_never_later_than_fifo():
    futs = _futs("pending", "pending", "pending")
    _later(futs[0][1], 0.1)   # the oldest finishes first: both return then
    waited = wr.bound_wait(futs, 2, True)
    assert 0.05 <= waited < 0.3
    for _t, f in futs[1:]:
        f.set_result(None)


def test_any_reraises_a_failed_collect():
    futs = _futs("pending", RuntimeError("W68 lane p0"), "pending")
    with pytest.raises(RuntimeError, match="W68"):
        wr.bound_wait(futs, 2, True)
    futs2 = _futs("pending", "pending", "pending")
    threading.Timer(0.05, lambda: futs2[1][1].set_exception(RuntimeError("W35"))).start()
    with pytest.raises(RuntimeError, match="W35"):
        wr.bound_wait(futs2, 2, True)
    for fl in (futs, futs2):
        for _t, f in fl:
            if not f.done():
                f.set_result(None)


def test_the_wake_loop_uses_it():
    src = (REPO / "python/flliper/srt/managers/scheduler_components/weight_updater.py").read_text()
    assert "_wake_futs[-(_n_wake_workers + 1)][1].result()" not in src
    assert "_pdflip_runahead.bound_wait(" in src and "_pdflip_runahead.runahead_any_on()" in src
    assert "runahead=%s runahead_wait_ms=%.0f" in src
