"""S18 (h16v3 boot ..._4c655e6fd1_1017_224250, D TP0 22:48:33-22:48:58): ``#1324 STORE READ INCOMPLETE`` on
weg2-1-14 (97344/118400), -15 (97344/118272), -16 (31168/58880) was NOT P's write-through lagging.

The store HELD the span (``TAIL-CONSUME ... hit=118400``, ``WEG2-SHORT-READ HELD ... the store holds the span``,
``W88-REROUTE store_held=118400``); D's read was cut by its own full arena (``WEG2-READ-DEPTH cause=arena_full``,
``READ-ROOM SHORT class=not_posted``: READ-ROOM serves group P only). The line nevertheless said "the store did not
hold the rest yet", so the S17 acceptance (STORE READ INCOMPLETE == 0) mixed two different defects.

Pinned on the metal numbers through the REAL ``Scheduler._weg2_note_store_shortfall``: a record whose probe hit
covers the deliverable prefix is named CAPACITY-CUT and counted apart; a record whose probe hit does not (P still
writing) keeps the old line; the deferral outcome is the same for both.
"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_weg2_store_short_fallback_rc12y as rc12y  # noqa: E402

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome

N, DELIVERED, DELIVERABLE, X = 118452, 97344, 118400, 12288
RID = "weg2-16-42"          # the rc12y harness's rid; the metal rid was weg2-1-14


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv(rc12y.MAX_CYCLES_ENV, raising=False)


def _run(hit_tokens, probed, caplog):
    s, r = rc12y._sched(N, DELIVERED, DELIVERABLE, X)
    s.tree_cache.prefetch_loaded_tokens_by_reqid[RID] = PrefetchOutcome(
        DELIVERED, hit_tokens=hit_tokens, probed=probed, matched=0, deliverable=DELIVERABLE, synced=DELIVERED)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        out = rc12y._cycle(s, r)
    return s, out, [rec.getMessage() for rec in caplog.records]


def test_a_read_the_store_held_is_named_a_capacity_cut(caplog):
    s, out, msgs = _run(DELIVERABLE, True, caplog)
    assert any("#1324 STORE READ CAPACITY-CUT" in m for m in msgs), msgs
    assert not any("#1324 STORE READ INCOMPLETE" in m for m in msgs), msgs
    assert getattr(s, "_weg2_store_short_capacity", 0) == 1
    assert out == "deferred"


def test_a_read_the_store_did_not_hold_keeps_the_store_lag_line(caplog):
    s, out, msgs = _run(65536, True, caplog)
    assert any("#1324 STORE READ INCOMPLETE" in m for m in msgs), msgs
    assert not any("CAPACITY-CUT" in m for m in msgs), msgs
    assert getattr(s, "_weg2_store_short_capacity", 0) == 0
    assert out == "deferred"


def test_an_unprobed_record_is_never_a_capacity_cut(caplog):
    _s, _out, msgs = _run(0, False, caplog)
    assert any("#1324 STORE READ INCOMPLETE" in m for m in msgs), msgs
    assert not any("CAPACITY-CUT" in m for m in msgs), msgs
