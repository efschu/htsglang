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


# ---- counter-tests: a decode stands -> the hold stays -----------------------------------------------------------------------


# ---- NF y6: the burst on an empty D ------------------------------------------------------------------------------------------


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
