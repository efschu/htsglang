"""X-SUM-PRICE on the arrival path (user 09.10.): "es soll doch gesammelt werden
und auf gesammelt geflipt werden".

NF y6 acceptance 09.10.: six SHORTs of 6239 uncached tokens arrived within one
second on an idle D; each passed X=12288 alone, D prefilled 37k tokens (~53 s),
and the front log held no DECODE-COLLECT line -- the rule opened only while D
DECODED. The book `DPrefillInflight` makes a D that PREFILLS a granted SHORT
busy, and carries its open tokens into the collected set's sum.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_prefill_inflight as dpf  # noqa: E402

X_LIVE = 12288


# ---- the book ---------------------------------------------------------------
def test_a_grant_counts_until_first_content():
    b = dpf.DPrefillInflight()
    b.grant(rid="a", tokens=6239, now=100.0)
    assert b.pending_tokens(now=101.0, live={}) == 6239
    b.enter_leg2(rid="a")
    assert b.pending_tokens(now=102.0, live={"a": 1.0}) == 6239
    b.done(rid="a")
    assert b.pending_tokens(now=103.0, live={"a": 1.0}) == 0


def test_a_leg2_row_dies_with_its_rid_on_d():
    b = dpf.DPrefillInflight()
    b.grant(rid="a", tokens=100, now=0.0)
    b.enter_leg2(rid="a")
    assert b.pending_tokens(now=1.0, live={}) == 0, "D no longer holds it (aborted / ended)"


def test_a_grant_that_never_reached_leg2_expires():
    b = dpf.DPrefillInflight()
    b.grant(rid="a", tokens=100, now=0.0)
    assert b.pending_tokens(now=dpf.GRANT_TTL_S - 1.0, live={}) == 100
    assert b.pending_tokens(now=dpf.GRANT_TTL_S + 1.0, live={}) == 0


# ---- the front ----------------------------------------------------------------
def _d_begins_every_grant(f):
    """S49 (SOLL R6, user 18.10. "Nur echt laufend zählt"): R6 counts a granted D prefill only once D began it. These
    burst tests model a D that begins every grant at once (the metal case: free seats, D extends ~80 ms after the
    POST); a D that has not begun it holds the set instead (test_pdflip_s49_r6_nur_echt_laufend_1018)."""
    from flliper.srt.pdflip import d_extend_beacon as _dxb
    def _read():
        # S56 (SOLL R6 + R11): D reports its REAL rest (here the granted tokens: nothing of them was cached) and the
        # legs can move to P (D-prefill legs in the lookahead) -- only then R6 counts them
        rows = f._d_pf_book()._rows
        f._rvp_state()
        for r in list(rows):
            f.__dict__.setdefault("_d_prefill_legs", {})[r] = time.time()
            f._leg2_lookahead.add(r)
        return {_dxb.rid_key(r): (time.time_ns(), int(rows[r].tokens)) for r in list(rows)}

    f._d_ext_reading = _read


def _on(monkeypatch, window="15"):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", window)
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)


def _idle_d_prefilling(f, tokens):
    """D holds nothing that decodes; one SHORT of `tokens` is granted, no first token."""
    f.tp_prefill_max_tokens = X_LIVE
    f._d_pf_book().grant(rid="first", tokens=tokens, now=time.time())


def test_the_default_is_on():
    assert envs.FLLIPER_PDFLIP_ENABLE_DECODE_COLLECT_PREFILL_BUSY.get() is True


def test_an_idle_d_with_nothing_in_flight_still_serves_at_once(monkeypatch):
    """The old rule stands: nothing prefilling, nothing decoding -> no window."""
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    assert f._dc_carried() == 0 and not f._dc_d_busy()
    assert asyncio.run(f._decode_collect_short("a", 6239)) is None
    assert f._dc_carried() == 6239, "served on D now: the gate booked it for the next arrival"


def test_a_d_that_prefills_a_granted_short_is_busy(monkeypatch):
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    _idle_d_prefilling(f, 6239)
    assert f._dc_carried() == 6239 and f._dc_d_busy()


def test_a_simultaneous_burst_books_before_the_seat_not_after(monkeypatch):
    """NF int23xsum metal run 09.10. (6 x 5200 on an idle D, X=6008): all six
    arrivals read the book in the same loop turn; the grant used to be booked
    only after the seat (the caller), so every one saw ``carried == 0``, took a
    D seat and D prefilled 31k tokens. No manual grant here -- the gate itself
    must book the first SHORT before the others look."""
    _on(monkeypatch, window="0.3")
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", "0")
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6008
    _d_begins_every_grant(f)

    async def burst():
        return await asyncio.gather(*[f._acquire_short_seat(f"s{i}", 5260, uncached=5200)
                                      for i in range(6)])

    seats = asyncio.run(burst())
    assert sum(s is not None for s in seats) == 1, "one SHORT on D, the sum 31200 > X sends the rest to P"
    assert f._d_pf_book().pending_tokens(now=time.time(), live={}) == 5200


def test_the_booking_stands_before_every_suspension(monkeypatch):
    """Review m3 (xsum-review-1009): the plain gather test stayed green when the
    booking moved BEHIND `await _d_seat.acquire()`, because no stub awaited
    between the gate and the seat. Here the awaits between them really suspend
    (the reading of D and the arrival-seat wait), as they do on metal: a booking
    taken after them lets every arrival of the burst read an empty book."""
    _on(monkeypatch, window="0.3")
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", "0")
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6008

    async def suspends(*a, **k):
        await asyncio.sleep(0.02)
        return None

    async def seat_wait(*a, **k):
        await asyncio.sleep(0.02)
        return True

    f._d_reading_if_armed = suspends
    f._arrival_seat_wait = seat_wait
    _d_begins_every_grant(f)

    async def burst():
        return await asyncio.gather(*[f._acquire_short_seat(f"s{i}", 5260, uncached=5200)
                                      for i in range(6)])

    seats = asyncio.run(burst())
    assert sum(s is not None for s in seats) == 1


def test_a_release_with_route_d_books_the_held_shorts(monkeypatch):
    """Review m2a: the set is priced as a set (3000 in flight + 2 x 500 <= X) and
    released to D; between that release and the caller's seat there are awaits,
    so the held SHORTs are booked in the release step itself."""
    _on(monkeypatch, window="0.3")
    monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_D_CHECK_S", "0")
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6008
    _idle_d_prefilling(f, 3000)

    async def two():
        return await asyncio.gather(f._decode_collect_short("a", 500), f._decode_collect_short("b", 500))

    out = asyncio.run(two())
    assert all(r in ("D", None) for r in out), out  # the release answers D, a late poller "serve now"; never P
    assert f._d_pf_book().pending_tokens(now=time.time(), live={}) == 3000 + 500 + 500


def test_switch_off_is_the_old_blind_path(monkeypatch):
    _on(monkeypatch, window="0.3")
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_DECODE_COLLECT_PREFILL_BUSY", "0")
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    _idle_d_prefilling(f, 6239)
    assert f._dc_carried() == 0
    assert asyncio.run(f._decode_collect_short("second", 6239)) is None, "straight to D, as before"
