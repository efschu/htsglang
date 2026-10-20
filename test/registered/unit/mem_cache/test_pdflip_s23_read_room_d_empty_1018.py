# SPDX-License-Identifier: Apache-2.0
"""S23 (18.10., h16v4b boot ..._3f26c7342d_1018_013102): READ-ROOM-D "greift nicht" -- what the metal actually said.

* 59x ``D-SPILL candidates=0 nodes=0`` with EVERY skip 0: the D tree held nothing a spill may take (ARENA-REF-HOLDERS on
  D at the same time: ``tree=0 own_held=0``; the arena was pinned by P's own trees, 46 P POSTs vs 1 D POST). A node
  WITHOUT a host copy was the one silent ``continue`` of ``form_a_spill``, so an empty tree and a tree of device-only
  nodes read alike. Now ``tree_nodes`` and ``no_host`` are in the line.
* 15x ``#1324 STORE READ CAPACITY-CUT`` on D (first one 01:36:24, pdflip-0-2 delivered=0): the READ-DEPTH cause of exactly
  those cuts was ``late_complete`` (``arena_full`` on D = 0 in the boot; D never posted for them). The label rested on
  "probe hit covers the deliverable" alone, which proves the STORE holds the span, not that the ARENA cut the read.
  CAPACITY-CUT now needs the recorded cause ``arena_full`` (or none recorded); a probe-covered cut with another cause is
  ``HELD-CUT cause=<c>``. Name only: the deferral is the same, rank-local cause, no rank decision (R11).
"""
import logging
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pdflip"))

import test_pdflip_read_room_d_1015 as RD  # noqa: E402,F401  (fixtures: roles, evict_mock)
import test_pdflip_store_short_fallback_rc12y as rc12y  # noqa: E402
from test_pdflip_read_room_d_1015 import _Pool, _real_d_tree, roles  # noqa: E402,F401
from test_pdflip_read_room_1015 import _rr  # noqa: E402

from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402

import threading  # noqa: E402
import pickle  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv(rc12y.MAX_CYCLES_ENV, raising=False)


# --- D-SPILL names what an unbacked / empty tree is ------------------------------------------------------------

def _spill_msgs(tree, caplog, need=10):
    rr = _rr()
    with caplog.at_level(logging.INFO, logger=rr.logger.name):
        got = rr.form_a_spill(tree, _Pool(None, {}, "/nonexistent"), need)
    return got, [r.getMessage() for r in caplog.records if "D-SPILL" in r.getMessage()]


def test_a_tree_of_device_only_nodes_is_named_no_host_not_silent(roles, caplog):
    roles.now[threading.current_thread().name] = "host"
    tree = _real_d_tree([(1, [0, 1, 2]), (1001, [3, 4, 5])], None, "tp0")
    for n in tree._t_nodes.values():
        n.component_data[RD.FULL].host_value = None     # D-written pages that never got a host copy
    got, msgs = _spill_msgs(tree, caplog)
    assert got == 0 and msgs, msgs
    assert "tree_nodes=2" in msgs[0] and "no_host:2" in msgs[0], msgs[0]
    assert "candidates=0" in msgs[0]


def test_an_empty_tree_says_tree_nodes_0(roles, caplog):
    roles.now[threading.current_thread().name] = "host"
    tree = _real_d_tree([], None, "tp0")
    got, msgs = _spill_msgs(tree, caplog)
    assert got == 0 and "tree_nodes=0" in msgs[0] and "no_host:0" in msgs[0], msgs


# --- CAPACITY-CUT only on arena_full -----------------------------------------------------------------------------

N, DELIVERED, DELIVERABLE, X = 118452, 97344, 118400, 12288
RID = "pdflip-16-42"


def _run(cut_cause, caplog):
    s, r = rc12y._sched(N, DELIVERED, DELIVERABLE, X)
    s.tree_cache.prefetch_loaded_tokens_by_reqid[RID] = PrefetchOutcome(
        DELIVERED, hit_tokens=DELIVERABLE, probed=True, matched=0, deliverable=DELIVERABLE, synced=DELIVERED,
        cut_cause=cut_cause)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        out = rc12y._cycle(s, r)
    return s, out, [rec.getMessage() for rec in caplog.records]


def test_late_complete_is_not_a_capacity_cut(caplog):
    s, out, msgs = _run("late_complete", caplog)      # the 01:36:24 metal cause
    assert any("#1324 STORE READ HELD-CUT" in m and "cut_cause=late_complete" in m for m in msgs), msgs
    assert not any("CAPACITY-CUT" in m for m in msgs), msgs
    assert getattr(s, "_pdflip_store_short_capacity", 0) == 0
    assert getattr(s, "_pdflip_store_short_held_other", 0) == 1
    assert out == "deferred"


def test_arena_full_is_a_capacity_cut(caplog):
    s, out, msgs = _run("arena_full", caplog)
    assert any("#1324 STORE READ CAPACITY-CUT" in m for m in msgs), msgs
    assert not any("HELD-CUT" in m for m in msgs), msgs
    assert getattr(s, "_pdflip_store_short_capacity", 0) == 1
    assert out == "deferred"


def test_no_recorded_cause_keeps_the_s18_capacity_label(caplog):
    s, out, msgs = _run(None, caplog)
    assert any("#1324 STORE READ CAPACITY-CUT" in m for m in msgs), msgs
    assert getattr(s, "_pdflip_store_short_capacity", 0) == 1 and out == "deferred"


def test_the_cause_survives_pickle_and_the_tree_builds_the_record_with_it():
    o = PrefetchOutcome(5, hit_tokens=9, probed=True, matched=1, deliverable=9, synced=5, cut_cause="join_unfinished")
    assert pickle.loads(pickle.dumps(o)).cut_cause == "join_unfinished"
    assert PrefetchOutcome(3).cut_cause is None
    from flliper.srt.mem_cache import unified_radix_cache as urc

    class Op:
        _pdflip_read_cut = (79, "late_complete")

    assert urc._s23_cut_cause(Op()) == "late_complete"
    assert urc._s23_cut_cause(object()) is None


@pytest.mark.parametrize("cause", ["l2_evicted_before_ref", "fill_evicted_before_ref"])
def test_eviction_clock_causes_are_capacity_cuts(cause, caplog):
    """The eviction clock took a page the read had found / landed: arena room pressure, not a HELD-CUT."""
    s, out, msgs = _run(cause, caplog)
    assert any("#1324 STORE READ CAPACITY-CUT" in m for m in msgs), msgs
    assert not any("HELD-CUT" in m for m in msgs), msgs
    assert getattr(s, "_pdflip_store_short_capacity", 0) == 1 and out == "deferred"


@pytest.mark.parametrize("cause", ["late_complete", "join_unfinished", "inflight_unfinished", "read_failed",
                                   "not_on_disk", "terminated", "arena_chain_recheck", "fill_none"])
def test_the_other_causes_are_held_cuts_and_the_text_does_not_claim_room(cause, caplog):
    s, out, msgs = _run(cause, caplog)
    held = [m for m in msgs if "#1324 STORE READ HELD-CUT" in m]
    assert held and f"cut_cause={cause}" in held[0], msgs
    assert "NOT by the arena room" not in held[0] and "CAPACITY-CUT" not in held[0], held[0]
    assert out == "deferred"
