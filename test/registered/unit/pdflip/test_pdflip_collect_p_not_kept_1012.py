"""DECODE-COLLECT's route=P is final: a released SHORT is queued BATCH, never SHORT-KEPT for D.

NF int24g 12.10. 21:01:07Z (front log): four SHORTs (174 + 4953 + 1457 + 5140 = 11724 uncached)
were released ``route=P`` (sum over the live X=8095), and in the same millisecond every one of them
printed ``SHORT-KEPT ... waits for a D seat ... no leg 1 on P`` and was prefilled by D
(``LEG2-FIRST-CONTENT via=d_direct``). A fifth request (7871, under X alone) found
``st["route"] == "P"`` still set for the phase, took the same exit and went to D too:
D TP0 prefilled 19595 = 11724 + 7871 tokens in 31 s. Cause: ``_acquire_short_seat`` returned
``None`` for the sum rule's "P" exactly as it does for a gate that held or a phase that moved,
and the handler's SK (#243) turns every such ``None`` with a measured presence into a keep.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import _front  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import front as F  # noqa: E402

X_LIVE = 8095


def _collecting(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_HALF_FLIP_S", "0.3")
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)


def _d_prefilling(tokens):
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    f._d_pf_book().grant(rid="first", tokens=tokens, now=time.time())
    # rc14 (S49, SOLL R6, user 18.10. "Nur echt laufend zählt"): "in flight" means D began the extend -- D's
    # D-EXTEND-BEACON says so; only then does R6 count it (a D that has not begun it holds the set instead,
    # test_pdflip_s49_r6_nur_echt_laufend_1018). Same fixture form as S49's _d_begins_every_grant.
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
    return f


def test_a_sum_over_x_tells_the_caller_it_is_the_sum_rules_p(monkeypatch):
    """11724 in flight + 7871 arriving > X=8095: the seat is refused and the refusal is NAMED
    ``collect_p`` -- the caller cannot tell it from a held gate otherwise."""
    _collecting(monkeypatch)
    f = _d_prefilling(5140)
    refused: list = []
    seat = asyncio.run(f._acquire_short_seat("second", 7900, refused, uncached=7871))
    assert seat is None
    assert refused == ["collect_p"]


def test_a_set_that_fits_is_served_by_d_and_carries_no_marker(monkeypatch):
    """5140 in flight + 800 arriving <= X: D takes it. ``collect_p`` must not degrade to
    always-true (that would send every collected SHORT to P)."""
    _collecting(monkeypatch)
    f = _d_prefilling(5140)
    refused: list = []
    seat = asyncio.run(f._acquire_short_seat("second", 900, refused, uncached=800))
    assert seat is not None
    assert refused == []


def test_a_later_single_arrival_after_the_release_is_its_own_class_not_the_phases(monkeypatch):
    """SOLL v2 R7 (the 663 case, inverted): there is no epoch rule. After the set's release a later arrival priced
    under X ALONE (800, kurz) is D's (R2) -- it is NOT named ``collect_p``; the committed rids are the set's alone."""
    _collecting(monkeypatch)
    f = _d_prefilling(5140)
    first: list = []
    assert asyncio.run(f._acquire_short_seat("second", 7900, first, uncached=7871)) is None
    assert "second" in f._set_st()["go_p"] and first == ["collect_p"]
    later: list = []
    seat = asyncio.run(f._acquire_short_seat("third", 900, later, uncached=800))
    assert seat is not None, "kurz: D at once"
    assert later == [] and "third" not in f._set_st()["go_p"]


def test_a_phase_that_moved_while_waiting_is_not_the_sum_rules_verdict(monkeypatch):
    """``_decode_collect_short`` also answers "P" when the epoch changed under the waiting SHORT. That
    is the old SK fall-through, not a sum verdict: no ``collect_p`` (the new epoch's state has no route)."""
    _collecting(monkeypatch)
    f = _d_prefilling(5140)

    async def moved(rid, unc, *a):
        f.epoch += 1
        return "P"

    f._decode_collect_short = moved
    refused: list = []
    assert asyncio.run(f._acquire_short_seat("second", 5100, refused, uncached=5000)) is None   # halb: it joins the set
    assert refused == []


def test_the_handler_consumes_the_marker_before_sk_decides():
    """``handle_generate`` is not unit-drivable; pin the ORDER that carries the fix: the marker is
    read and removed from ``short_refused`` (d_eligible / x_deferred keep their meaning) BEFORE
    ``_sk`` is computed, and ``_sk`` excludes it."""
    src = inspect.getsource(F.Front.handle_generate)
    read = src.index('_collect_p = "collect_p" in short_refused')
    drop = src.index('short_refused.remove("collect_p")')
    sk = src.index("_sk = (short_ok and _short_keep_enabled()")
    assert read < drop < sk
    assert "and not _collect_p)" in src[sk:sk + 400]


# ---------------------------------------------------------------------------------------------
# Behaviour through the REAL handler (planer-collectp-1012): the source-order pin above says where
# the fix sits; these say what the request does. Real Front, real _acquire_short_seat and
# DECODE-COLLECT; only the group RPCs, the leg-2 forward and the pricing's presence reading are
# stubbed (the same approach as test_pdflip_early_flip_1002). SK (#243) is armed by the stubbed
# pricing: a measured presence (``store_span`` > 0, src ``d_leg2_cached``) is exactly what let the
# handler turn the collect gate's ``None`` into SHORT-KEPT on 12.10.
# ---------------------------------------------------------------------------------------------
import logging  # noqa: E402
import re  # noqa: E402

X_H = 4096


class _Req:
    def __init__(self, payload, path="/v1/messages"):
        self._p = payload
        self.path = path

    async def json(self):
        return self._p


def _payload(uncached):
    # the stubbed pricing reads the uncached price out of the text (U<n>)
    return {"model": "m", "max_tokens": 2, "messages": [{"role": "user", "content": "U%d " % uncached + "a" * 40}]}


def _handler_front(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_HALF_FLIP_S", "0.4")
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_PRICE_BARRIER", raising=False)

    def _price(text, spans, epoch=None):
        return int(re.search(r"U(\d+) ", text).group(1)), 60000, True  # (uncached, prompt, presence known)

    monkeypatch.setattr(F, "price_remainder", _price)
    f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="collectp",
                store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                weight_chunks=2, tp_prefill_max_tokens=X_H, flip_min_work_tokens=X_H)
    f.state = "serving"
    f.groups["D"].outstanding["running-1"] = time.time()  # D decodes: the arrivals collect
    f.d_legs, f.p_legs = [], []
    f.spans.last_src = "d_leg2_cached"

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.d_legs.append(rid)
        return F.web.json_response({})

    async def leg1(p):
        f.p_legs.append(p.rid)

    async def rpc(g, path, body, timeout):
        return 200, "{}"

    f.leg2, f.leg1, f.rpc = leg2, leg1, rpc
    f._kick_controller = lambda *a, **k: None  # the controller loop is not running: nothing flips
    f._arm_client_watch = lambda *a, **k: None
    return f


async def _drive(f, uncached_list, settle=1.4, stagger=0.0, caplog=None):
    tasks = []
    for u in uncached_list:
        tasks.append(asyncio.create_task(f.handle_generate(_Req(_payload(u)))))
        if stagger:
            await asyncio.sleep(stagger)
    await asyncio.sleep(settle)
    for t in tasks:
        if not t.done():
            t.cancel()
    res = await asyncio.gather(*tasks, return_exceptions=True)
    for r in res:  # a handler that crashed would read as "nothing happened": surface it
        if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
            raise r


def _d_direct_prefill_sum(f):
    """What D is committed to prefill by D-direct handoff: everything marked d_direct, wherever it
    waits (D's admission line, or already taken by a D leg 2)."""
    return sum(int(p.est_uncached) for p in list(f._ready_for_d) if getattr(p, "d_direct", False))


def test_handler_set_over_x_is_queued_for_p_not_kept_for_d(monkeypatch, caplog):
    """12.10. 21:01:07: four SHORTs (sum over X) released route=P. Each must sit in the P queue
    (BATCH, leg 1 on P), none in D's admission line, none SHORT-KEPT."""
    f = _handler_front(monkeypatch)
    sizes = [3000, 3000, 3000]  # each under X alone, 9000 together > X=4096
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_drive(f, sizes))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("SET-VERDICT" in m and "verdict=FLIP_P" in m for m in msgs), msgs
    assert not [m for m in msgs if m.startswith("PDFLIP SHORT-KEPT")], "released-to-P SHORT was kept for D"
    assert len([m for m in msgs if m.startswith("PDFLIP COLLECT-P-QUEUED")]) == 3
    assert len(f.queue) == 3 and all(not p.skip_leg1 and not p.short_kept and not p.d_direct for p in f.queue)
    # the refusal marker is consumed, not left in short_refused: the entries keep d_eligible (the sum-priced
    # drains still see them), they are only not kept for D
    assert all(p.d_eligible for p in f.queue)
    assert len(f._ready_for_d) == 0 and f.d_legs == []
    # the Summenregel invariant: whatever D is handed directly stays within X
    assert _d_direct_prefill_sum(f) <= X_H
    assert f.counters["collect_p_not_kept"] == 3 and f.counters["short_kept_d"] == 0


def test_handler_single_kurz_request_after_a_p_release_is_served_by_d(monkeypatch, caplog):
    """SOLL v2 R7 inverts the 663 case: after the release (route P for the three halb SHORTs) a later arrival
    priced UNDER halbX alone (900, kurz) is D's -- no epoch rule sends it to P."""
    f = _handler_front(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_drive(f, [3000, 3000], settle=1.0))  # the pair flips: FLIP_P
        asyncio.run(_drive(f, [900], settle=0.5))  # one kurz request, same epoch
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if m.startswith("PDFLIP SHORT-KEPT")], msgs
    assert len(f.queue) == 2 and f.d_legs and [int(p.est_uncached) for p in f.queue].count(900) == 0
    assert any("verdict=D_NOW why=R2" in m and "classes=kurz" in m for m in msgs)


def test_handler_a_kurz_pair_is_served_by_d_at_once(monkeypatch, caplog):
    """Control: requests under halbX are D's at once (R2) -- no COLLECT-P-QUEUED, no wait."""
    f = _handler_front(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_drive(f, [1000, 1000]))
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "COLLECT-P-QUEUED" in m]
    assert len(f.d_legs) == 2 and len(f.queue) == 0
    assert f.counters["collect_p_not_kept"] == 0


def test_handler_a_halb_alone_is_served_by_d_after_t_half(monkeypatch, caplog):
    """Control for R3: one halb SHORT waits T_halb (0.4 s here), then D takes it; no flip, no COLLECT-P-QUEUED."""
    f = _handler_front(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_drive(f, [3000]))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("SET-VERDICT" in m and "verdict=D_AFTER_TIMER" in m for m in msgs), msgs
    assert not [m for m in msgs if "COLLECT-P-QUEUED" in m]
    assert len(f.d_legs) == 1 and len(f.queue) == 0 and f.counters["collect_p_not_kept"] == 0


def test_handler_sk_still_keeps_a_gate_held_short(monkeypatch, caplog):
    """The fix must not break SK for what it is for: a SHORT whose seat path ended in ``None`` for
    another reason (here: D not serving any more -> falls through; SK keeps it only while D is awake
    -- so use the held batch gate: it times out into BATCH and SK queues it D-direct)."""
    f = _handler_front(monkeypatch)
    f.drain_deadline_s = 0.2
    f._batch_gate.clear()  # BATCH work holds the gate: the SHORT seat path times out into None
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(_drive(f, [900], settle=1.0))
    msgs = [r.getMessage() for r in caplog.records]
    assert [m for m in msgs if m.startswith("PDFLIP SHORT-KEPT")], msgs
    assert not [m for m in msgs if "COLLECT-P-QUEUED" in m]
    assert f.counters["short_kept_d"] == 1 and f.counters["collect_p_not_kept"] == 0
