# SPDX-License-Identifier: Apache-2.0
"""DUAL P-KV MAP-SHORT WAIT (4a) and the full-gap booking (4b).

Metal gmps7 (boot dkr27bnvfp4dual1mbar1fs10011748, 17:53:15-16): D retracted four
decodes, the four W50 re-routes made PP0's GRANT-SUM 151552 tokens; the card
ledger granted it, the card was physically short (``LEDGER-PHYS OVER-PROMISE by
465567744 B``; the reconciler had booked only the window MINIMUM 208994304 B) and
``_move``'s ``Weg2DualKvMapShort`` killed PP0 (``tms_set_spans rc=2 (cuMemCreate
out of memory) moving v3 to 151552 tokens``).

DANGER DIRECTIONS guarded here:
* (4a) a physically short card is a WAIT on PP0: map_granted rolls back to the
  standing mapping and adopts nothing, pp0_grant returns every card's charge and
  answers 0 (the intake holds the request) -- never a raise out of the grant;
* (4b) a persistent over-promise books the gap measured NOW, not the smallest of
  the window.
MUTANT per guard (asserted in-suite on the real source): the catch removed ->
the short raises again; the window minimum booked -> the ledger still over-promises.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

G = 2 << 20
ALLOC = (65536 + 64) * 2048
MIB = 1 << 20


class ShortSpans:
    """tms_set_spans stand-in: rc=2 (cuMemCreate OOM) for any plan mapping more
    than ``cap`` bytes; smaller plans (a rollback) succeed."""

    def __init__(self, cap=None):
        self.cap = cap
        self.calls = []

    def set_spans(self, ptr, spans, now):
        size = sum(e - s for s, e in spans)
        self.calls.append(size)
        return 2 if self.cap is not None and size > self.cap else 0


def _geom():
    return S._geom_for(torch.zeros(16448, 512), 16384, 64, "k", ALLOC)


def _card(budget_mib=400):
    path = os.path.join(tempfile.mkdtemp(prefix="wkvms"), "card")
    d = K.CardKvLedger(path, "D")
    d.contribute(budget_mib * MIB, committed=0)
    led = K.CardKvLedger(path, "P")
    led.contribute(0)
    return path, led


def _stage(led, spans):
    return S.PKvStage([(1, _geom())], led, allocator=object(), pools=[], page_size=64, granule=G,
                      top_tokens=16384, spans=spans, engage_cap=lambda *a: None)


def _plan_bytes(st, tokens):
    return sum(e - s for s, e in st._plan(st.born[0][1], tokens))


# -- (4a) map_granted rolls back ---------------------------------------------


def test_a_short_card_rolls_the_mapping_back_and_adopts_nothing():
    _path, led = _card()
    spans = ShortSpans()
    st = _stage(led, spans)
    led.request(st.bytes_for(4096) - st.bytes_for(0))     # PP0's charge for the first grant
    st.map_granted(4096)
    committed = st._committed
    spans.cap = _plan_bytes(st, 8192)                     # the card holds 8192 tokens' spans, not 12288
    with pytest.raises(S.Weg2DualKvMapShort):
        st.map_granted(12288)
    assert st.mapped_tokens == 4096, "the standing mapping is kept"
    assert spans.calls[-1] == _plan_bytes(st, 4096), "the last move maps back exactly the standing level"
    assert st._committed == committed, "nothing adopted for a mapping that did not happen"


# -- (4a) pp0_grant holds the request ------------------------------------------


def _pp0(monkeypatch, tmpdir, cap_tokens, pp0_grant=None):
    path, led = _card()
    spans = ShortSpans()
    st = _stage(led, spans)
    spans.cap = _plan_bytes(st, cap_tokens)
    tag = "mapshort-%s" % os.path.basename(str(tmpdir))
    monkeypatch.setenv("SGLANG_WEG2_DUAL_KV_TAG", tag)
    sf = S.publish_stage(st, tag, 0)
    sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
        **{S.ACTOR_ATTR: st})), ps=types.SimpleNamespace(pp_rank=0, pp_size=1))
    req = types.SimpleNamespace(rid="weg2-0-2", origin_input_ids=list(range(10000)))
    try:
        lvl = (pp0_grant or S.pp0_grant)(sched, req)
    finally:
        os.unlink(sf)
    return lvl, st, req, path


def test_pp0_grant_on_a_physically_short_card_waits_instead_of_raising(monkeypatch, tmp_path):
    lvl, st, req, path = _pp0(monkeypatch, tmp_path, cap_tokens=8192)   # 10000 tokens -> 12288 level
    assert lvl == 0, "a short card holds the request (the intake's 0), it does not kill PP0"
    assert st.mapped_tokens == 0
    assert not getattr(req, "_dual_kv_tokens", 0)
    assert K.peek(path).committed["P"] == 0, "every card's charge went back"


def test_pp0_grant_still_grants_when_the_card_has_the_bytes(monkeypatch, tmp_path):
    lvl, st, req, path = _pp0(monkeypatch, tmp_path, cap_tokens=16384)
    assert lvl == 12288 and st.mapped_tokens == 12288 and req._dual_kv_tokens == 12288
    assert K.peek(path).committed["P"] >= st.bytes_for(12288) - st.bytes_for(0)


def _mutant(fn, fixed, back):
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant"
    ns = dict(vars(S))
    exec(compile(src.replace(fixed, back), S.__file__, "exec"), ns)
    return ns[fn.__name__]


def test_the_no_catch_mutant_turns_the_wait_test_red(monkeypatch, tmp_path):
    m = _mutant(S.pp0_grant, "except Weg2DualKvMapShort as exc:", "except ZeroDivisionError as exc:")
    with pytest.raises((AssertionError, S.Weg2DualKvMapShort)):
        lvl, *_ = _pp0(monkeypatch, tmp_path, cap_tokens=8192, pp0_grant=m)
        assert lvl == 0


# -- (4b) the full current gap is booked -----------------------------------------


def _run_phys(monkeypatch, overs_mib, phys_check=None):
    path, led = _card(budget_mib=4000)
    actor = types.SimpleNamespace(ledger=led)
    clock = [0.0]
    monkeypatch.setattr(S, "_now", lambda: clock[0])
    free = K.peek(path).free
    for o in overs_mib:
        monkeypatch.setattr(S, "phys_free_bytes", lambda o=o: free - o * MIB)
        (phys_check or S.phys_check)(actor, "D")
        clock[0] += S.PHYS_CHECK_S
    return path, free


def test_a_persistent_over_promise_books_the_gap_measured_now(monkeypatch):
    path, free = _run_phys(monkeypatch, [209, 477, 466])        # the metal window
    st = K.peek(path)
    assert free - st.free == 466 * MIB, "booked the current 466 MiB gap, not the window minimum 209"
    assert st.free == free - 466 * MIB


def test_a_single_spike_books_nothing(monkeypatch):
    path, free = _run_phys(monkeypatch, [466, 0, 466])
    assert K.peek(path).free == free, "persistence still decides WHETHER to book"


def test_the_window_minimum_mutant_turns_the_booking_test_red(monkeypatch):
    m = _mutant(S.phys_check, "n = led.reconcile(phys, cap=over)",
                "n = led.reconcile(phys, cap=min(win[-PHYS_BOOK_AFTER:]))")
    path, free = _run_phys(monkeypatch, [209, 477, 466], phys_check=m)
    assert free - K.peek(path).free != 466 * MIB


# -- residual risk: a follower short is a NAMED stop -------------------------------


def test_a_follower_that_cannot_map_pp0s_grant_stops_named_and_rolled_back():
    _path, led = _card()
    spans = ShortSpans()
    st = _stage(led, spans)
    spans.cap = _plan_bytes(st, 8192)
    sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
        **{S.ACTOR_ATTR: st})))
    item = types.SimpleNamespace(rid="weg2-0-2", **{S.WIRE_DUAL_KV: 12288})
    with pytest.raises(S.Weg2DualKvFollowerMapShort, match="DUAL-FOLLOWER-MAP-SHORT"):
        S.on_told(sched, item)
    assert st.mapped_tokens == 0
