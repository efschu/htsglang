"""DECODE-COLLECT NODECODE 1015 (user law, core function): collecting (the hold up to d_check_s / window_s) is allowed ONLY while
a DECODE stands on D. With no decode the set is decided AT ONCE: its sum (X-SUM-PRICE: pending + carried) fits under X -> D,
else P. X-SUM-PRICE itself is unchanged -- ``carried`` still counts into the sum; only the wait without a decode goes.

Defect 1 (report deskq/done/nf-core-collect-1015.md, sec. 5 + round 2 sec. 4): ``_decode_collect`` used ``_dc_d_busy()`` (=
decoding OR carried > 0) as "decoding", and ``_dc_decoding()`` counted a rid D still PREFILLS (in D's ledger, open row of the
X-SUM-PRICE book, no first content) as a decode. Front-ledger replay of the 10 D-log "carried holders" of 15.10.
(/tmp/nodecode1015/ledger.py): 8 had real decodes standing (first content seen, still streaming -- the hold was lawful, D's
rounds only stalled behind the prefill chunks); 2 had NOTHING on D but the carried prefill:

* boot 17c2563647_1015_042125 front.log :37843 06:38:41,240 D-ADMIT weg2-86-985 (SHORT, 2265 uncached) seat=1/6 (seats_free=6
  at :37832), :37847 X-ROUTE weg2-86-986 est_uncached=307 X=5760, :37848 06:38:41,256 ``hold n=1 tokens=2572 X=5760``, :37863
  06:38:48,756 release waited_s=7.5 route=D stage=dcheck.
* boot 17c2563647_1015_070103 front.log :1297 07:08:13,421 D-ADMIT weg2-4-14 (1488 uncached, :1263) seat=1/6, weg2-4-15 granted
  (353, :1276; D-ADMIT only :1310 07:08:14,432), :1306 X-ROUTE weg2-4-16 252 X=5060, :1307 07:08:13,651 ``hold n=1 tokens=2093
  X=5060``, :1319 07:08:21,197 release waited_s=7.5 route=D stage=dcheck.

Defect 2 (same report, "einziger Rest" 14.10. 16:39:02Z boot fe799c4d6e): the two W50-REROUTE midstream streams (weg2-10-71,
weg2-9-69) are in ``_rvp_inflight`` (front.py ``_rvp_take``) and ``_flip_ledger`` drops them -- they never counted as a decode.
The ledger at the hold (front :3590-:3592 16:38:56) still held weg2-6-60/6-67/6-68 (resumed decodes, PARK-RESUME :3555) and
weg2-10-73 (hand-off): D TP2 had died at 16:38:51 (D.log :63949-63952 ``gloo::EnforceNotMet ... Fatal Python error: Aborted``),
the front learned it from the broken streams at 16:39:02,252 and released 68 ms later. Pinned here: the RVP exclusion (both ways).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _pending  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as front_mod  # noqa: E402


def _on(monkeypatch, window="15"):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)  # default 7.5
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", window)
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY", raising=False)  # default on


def _prefilling(f, rid, tokens, *, leg2=True):
    """The front's book for a SHORT D prefills: grant at the verdict, ``enter_leg2`` when D holds it (D-ADMIT)."""
    book = f._d_pf_book()
    book.grant(rid=rid, tokens=tokens, now=time.time())
    if leg2:
        book.enter_leg2(rid=rid)


def _releases(caplog):
    return [r.getMessage() for r in caplog.records if "DECODE-COLLECT release" in r.getMessage()]


def _holds(caplog):
    return [r.getMessage() for r in caplog.records if "DECODE-COLLECT hold" in r.getMessage()]


# ---- the two real cases of 15.10. --------------------------------------------------------------------------------------------

def _front_0638(monkeypatch):
    """06:38:41,256: D holds weg2-86-985 alone (D-ADMIT :37843, 2265 uncached, no first content) -- no decode."""
    _on(monkeypatch)
    f = _front(running=["weg2-86-985"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 5760
    _prefilling(f, "weg2-86-985", 2265)
    return f


def test_real_0638_no_decode_the_sum_goes_to_d_at_once(monkeypatch, caplog):
    f = _front_0638(monkeypatch)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("weg2-86-986", 307, t0)]
    assert not f._dc_decoding(), "D only prefills weg2-86-985: no decode stands"
    assert f._dc_d_busy(), "the carried prefill still prices the set (X-SUM-PRICE)"
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(t0) == "D", "base: 'hold' for 7.5 s (front :37848 -> :37863)"
    (line,) = _releases(caplog)
    assert "n=1 tokens=2572 X=5760 waited_s=0.0 route=D stage=d-idle p_bound=False" in line, line
    assert not _holds(caplog) and f.counters["decode_collect_hold"] == 0


def test_real_0708_no_decode_the_sum_goes_to_d_at_once(monkeypatch, caplog):
    """07:08:13,651: weg2-4-14 (1488) on D prefilling, weg2-4-15 (353) granted but not yet admitted, arrival weg2-4-16 (252):
    2093 <= X=5060 -> D now (base: hold 7.5 s, :1307 -> :1319)."""
    _on(monkeypatch)
    f = _front(running=["weg2-4-14"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 5060
    _prefilling(f, "weg2-4-14", 1488)
    _prefilling(f, "weg2-4-15", 353, leg2=False)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("weg2-4-16", 252, t0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(t0) == "D"
    (line,) = _releases(caplog)
    assert "n=1 tokens=2093 X=5060 waited_s=0.0 route=D stage=d-idle" in line, line


def test_no_decode_and_the_sum_over_x_goes_to_p_at_once(monkeypatch, caplog):
    """Counter-test, the other branch of the law: the same D (only 2265 carried) and an arrival of 4000 -> 6265 > X=5760 -> P now,
    not after 15 s. The sum includes ``carried`` (4000 alone would fit)."""
    f = _front_0638(monkeypatch)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("big", 4000, t0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(t0) == "P"
    (line,) = _releases(caplog)
    assert "tokens=6265 X=5760 waited_s=0.0 route=P stage=d-idle" in line, line


# ---- counter-tests: a decode stands -> the hold stays -----------------------------------------------------------------------

def test_real_0744_a_decode_stands_so_the_set_collects(monkeypatch, caplog):
    """07:44:45,728 (boot 17c2563647_1015_073325 front :2755 hold n=1 tokens=7179 X=6530): D's ledger held weg2-8-45 DECODING
    (first content :2548 07:43:59,566, served :2787 07:45:02) next to weg2-8-51/52/53 prefilling (5763 carried, released :2724).
    The D log showed no round because the chunks stalled them -- but a decode stood: the hold is lawful and stays."""
    _on(monkeypatch)
    f = _front(running=["weg2-1-6", "weg2-8-45", "weg2-8-34", "weg2-8-53", "weg2-8-51", "weg2-8-52"], n=6,
               kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6530
    # X-ROUTE est_uncached :2663/:2673/:2695 (admitted :2735-:2740), :2714/:2722 (granted at the release :2724, not admitted yet)
    for rid, u in (("weg2-8-53", 1262), ("weg2-8-51", 2061), ("weg2-8-52", 2184)):
        _prefilling(f, rid, u)
    for rid, u in (("weg2-8-54", 180), ("weg2-8-55", 36)):
        _prefilling(f, rid, u, leg2=False)
    # synthetic: 7179 - 1416 - 5723 = 40 carried the log does not attribute; weg2-8-34 (:2227) is the admitted SHORT with no
    # first-content line before the hold
    _prefilling(f, "weg2-8-34", 40)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("weg2-8-56", 1416, t0)]
    assert f._dc_decoding(), "weg2-8-45 (and weg2-1-6, first content :1470) decode"
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(t0) == "hold"
        assert f._decode_collect(t0 + 7.6) == "hold", "7179 > X: dcheck holds on while the decode stands"
    (h,) = _holds(caplog)
    assert "n=1 tokens=7179 X=6530 waited_s=0.0" in h, h


def test_a_decode_only_d_still_holds_to_d_check(monkeypatch):
    """Plain decodes (no book row): the old window is untouched -- hold, then D at d_check_s=7.5."""
    _on(monkeypatch)
    f = _front(running=["r1", "r2"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6057
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("a", 213, t0)]
    assert f._decode_collect(t0 + 0.1) == "hold"
    assert f._decode_collect(t0 + 7.0) == "hold"
    assert f._decode_collect(t0 + 7.5) == "D"


def test_the_last_decode_ends_while_carried_remains_releases_at_once(monkeypatch, caplog):
    """13:12:55 shape (test_weg2_collect_busy_d_1015): five decodes + weg2-40-128 carried (5089), arrival 213. The decodes end at
    +2.0 s while 40-128 still prefills: no decode stands any more -> release now with the sum 5302 (base: held on to 7.5 s,
    because ``carried > 0`` kept ``_dc_d_busy`` True)."""
    _on(monkeypatch)
    f = _front(running=["r1", "r2", "r3", "r4", "r5", "weg2-40-128"], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 6057
    _prefilling(f, "weg2-40-128", 5089)
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("weg2-42-129", 213, t0)]
    with caplog.at_level(logging.INFO, logger="weg2.front"):
        assert f._decode_collect(t0 + 0.1) == "hold"
        for r in ("r1", "r2", "r3", "r4", "r5"):
            f.groups["D"].outstanding.pop(r)
        assert f._decode_collect(t0 + 2.0) == "D"
    (line,) = _releases(caplog)
    assert "n=1 tokens=5302 X=6057 waited_s=2.0 route=D stage=d-idle" in line, line


# ---- NF y6: the burst on an empty D ------------------------------------------------------------------------------------------

def test_y6_burst_over_x_on_an_empty_d_goes_to_p_not_d_and_at_once(monkeypatch):
    """NF y6 09.10.: six SHORTs of 6239 in one second on an idle D, X=12288. The first is booked (carried 6239); the second sees
    6239 + 6239 = 12478 > X -> P (sticky for the phase); the rest follow to P. Never all six on D (37k), and no 15 s wait."""
    _on(monkeypatch)  # window 15 s, d_check 7.5 s: a hold would show in the clock
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 12288

    async def burst():
        return await asyncio.gather(*[f._acquire_short_seat(f"s{i}", 6300, uncached=6239) for i in range(6)])

    t0 = time.time()
    seats = asyncio.run(burst())
    took = time.time() - t0
    assert sum(s is not None for s in seats) == 1, "one SHORT on D, the sum sends the rest to P"
    assert f.counters["decode_collect_release_P"] >= 1 and f.counters["decode_collect_release_D"] == 0
    assert f.counters["decode_collect_hold"] == 0, "no decode on D: no hold"
    assert took < 2.0, f"decided at once, not after the window ({took:.1f} s)"
    assert f._dc_st()["route"] == "P"


def test_y6_burst_under_x_stays_on_d_at_once(monkeypatch):
    """Control: two 3000-token SHORTs on an empty D (sum 6000 <= X=12288): both on D, no wait."""
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    f.tp_prefill_max_tokens = 12288

    async def burst():
        return await asyncio.gather(*[f._acquire_short_seat(f"s{i}", 3100, uncached=3000) for i in range(2)])

    t0 = time.time()
    seats = asyncio.run(burst())
    assert sum(s is not None for s in seats) == 2
    assert time.time() - t0 < 2.0 and f.counters["decode_collect_hold"] == 0
    assert f._d_pf_book().pending_tokens(now=time.time(), live={}) == 6000


# ---- what counts as a decode -------------------------------------------------------------------------------------------------

def test_a_mixed_ledger_counts_the_decode_not_the_prefill(monkeypatch):
    """One rid decodes, one prefills: a decode stands (the exception is per rid, not all-or-nothing)."""
    _on(monkeypatch)
    f = _front(running=["dec", "pf"], n=6, kv={"available": 400000, "evictable": 0})
    _prefilling(f, "pf", 900)
    assert f._dc_decoding()
    f.groups["D"].outstanding.pop("dec")
    assert not f._dc_decoding()


def test_first_content_turns_the_prefill_into_a_decode(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["weg2-86-985"], n=6, kv={"available": 400000, "evictable": 0})
    _prefilling(f, "weg2-86-985", 2265)
    assert not f._dc_decoding()
    f._d_pf_book().done(rid="weg2-86-985")  # the front's hook at LEG2-FIRST-CONTENT
    assert f._dc_decoding()


def test_switch_off_keeps_the_ledger_alone(monkeypatch):
    """SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY=0: no book in the decision, the ledger alone (as before)."""
    _on(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY", "0")
    f = _front(running=["weg2-86-985"], n=6, kv={"available": 400000, "evictable": 0})
    _prefilling(f, "weg2-86-985", 2265)
    assert f._dc_decoding()


# ---- defect 2: RESUME-VIA-P streams are no decode (pin, both ways) -----------------------------------------------------------

def _front_1638(monkeypatch, rvp):
    """16:38:56 (boot fe799c4d6e front :3590/:3591): weg2-10-71 and weg2-9-69 W50-REROUTE midstream -> ``_rvp_inflight``."""
    _on(monkeypatch)
    f = _front(running=["weg2-10-71", "weg2-9-69"], n=6, kv={"available": 400000, "evictable": 0})
    del f._flip_ledger  # the real ledger: parked AND RESUME-VIA-P rids out
    f._rvp_inflight = set(rvp)
    f.tp_prefill_max_tokens = 6056
    return f


def test_rerouted_midstream_streams_are_no_decode(monkeypatch):
    f = _front_1638(monkeypatch, rvp=("weg2-10-71", "weg2-9-69"))
    assert front_mod.Front._flip_ledger(f, f.groups["D"]) == []
    assert not f._dc_decoding() and not f._dc_d_busy()
    t0 = time.time()
    f.t_awake = t0 - 60.0
    f.queue = [_pending("weg2-12-74", 1185, t0)]
    assert f._decode_collect(t0) is None, "nothing stands on D: no window"


def test_a_resumed_decode_beside_the_rerouted_ones_still_counts(monkeypatch):
    """Counter-test: weg2-6-60 (resumed at PARK-RESUME :3555, decoding) next to the two RESUME-VIA-P rids -> a decode stands."""
    f = _front_1638(monkeypatch, rvp=("weg2-10-71", "weg2-9-69"))
    f.groups["D"].outstanding["weg2-6-60"] = 1.0
    assert front_mod.Front._flip_ledger(f, f.groups["D"]) == ["weg2-6-60"]
    assert f._dc_decoding()
