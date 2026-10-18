"""H16Q review (nf-h16q-review-1015.md) A2 / A3 / A4 -- hermetic, no CUDA.

A2  W3-FORMA: a worker that cannot apply TP0's SPILL (short / anchor / deleted / unreconcilable) is named at EVERY
    occurrence (``R12 SPILL-DIVERGED``), not only counted in the rate-limited APPLIED line (law raenge-nie-uneins;
    the metal count must be 0, the group stop follows in the next batch if it is not).
A3  W3-FORMA: a FORMA-REQUEST hands at most SGLANG_WEG2_W3_FORMA_SPILL_MAX_PAGES (128) pages to the synchronous L3
    write on D's TP0 scheduler thread, and its line carries ``ms=``.
A4  B5 metal tripwire: a W27-UNIFORM FOLLOWER SHORT / HELD of a rid within 30 s after its TOLD-FLOOR is named
    (``PF TOLD-FLOOR-THEN-W27``), once per (rid, kind), no stop.

A2/A3 run on the dual15 fixture (the real C arena, the real ArenaMHAHostPool per rank on ONE arena file, the real
``_weg2_direct_claim`` / ``_w3_arena_spill`` / ``form_a_host_shadow`` attach + apply).
"""
import logging
import os
import re
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_w3_arena_spill_forma_dual15_0930 import FULL, _as, _claimer, group  # noqa: E402,F401

from sglang.srt.managers import weg2_store_told as ST  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as FB  # noqa: E402
from sglang.srt.mem_cache import form_a_host_shadow as R  # noqa: E402

CAP_ENV = "SGLANG_WEG2_W3_FORMA_SPILL_MAX_PAGES"


def _tp0_serves_a_worker_request(g):
    with _as("worker"):
        m1 = _claimer(g.t1, "b")
        assert g.t1._weg2_direct_claim(m1) is False
    with _as("host"):
        R.register_tree(g.t0)
        sent = R.attach([])
    assert sent and isinstance(sent[0], R.FormAHostVerdict)
    return sent[0]


# ------------------------------------------------------------------------------------------------ A2

def test_a2_a_spill_the_worker_cannot_apply_is_named_at_every_occurrence(group, monkeypatch, caplog):
    g = group
    monkeypatch.delenv(CAP_ENV, raising=False)
    v = _tp0_serves_a_worker_request(g)
    spilled_ids = {id(n) for n in g.n0 if n.component_data[FULL].host_value is None}
    i_both = [i for i, n in enumerate(g.n0) if id(n) in spilled_ids and g.n1[i].children]
    assert len(i_both) >= 2, "the fixture spilled fewer than two interior nodes"
    # this worker's copies of two interior nodes left its device: TP0's SPILL is unreconcilable here
    for i in i_both[:2]:
        g.n1[i].component_data[FULL].value = None
    caplog.set_level(logging.WARNING, logger=R.logger.name)
    with _as("worker"):
        c = R.apply(g.t1, v.events, seq=v.seq)
    assert c["unreconcilable"] == 2
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("R12 SPILL-DIVERGED")]
    assert len(lines) == 2, lines                                 # each occurrence, not a counter
    assert all("out=unreconcilable" in s and f"seq={v.seq}" in s and "node=" in s for s in lines)


def test_a2_a_clean_spill_names_nothing(group, monkeypatch, caplog):
    g = group
    monkeypatch.delenv(CAP_ENV, raising=False)
    v = _tp0_serves_a_worker_request(g)
    caplog.set_level(logging.WARNING, logger=R.logger.name)
    with _as("worker"):
        c = R.apply(g.t1, v.events, seq=v.seq)
    assert c["spill"] >= 1
    assert not [r for r in caplog.records if "SPILL-DIVERGED" in r.getMessage()]


# ------------------------------------------------------------------------------------------------ A3

def _forma_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("W3-ARENA FORMA-REQUEST")]


def test_a3_forma_request_secures_at_most_the_cap_and_names_ms(group, monkeypatch, caplog):
    g = group
    monkeypatch.setenv(CAP_ENV, "3")
    caplog.set_level(logging.INFO)
    v = _tp0_serves_a_worker_request(g)
    spilled = [n for n in g.n0 if n.component_data[FULL].host_value is None]
    assert len(spilled) == 3, "one page per node: the cap of 3 pages lets 3 nodes through the L3 write"
    assert len(list(g.store.glob("*.bin"))) <= 3, "more than the cap reached the synchronous L3 write"
    assert len([e for e in v.events if e[0] == R.SPILL]) == 3
    (line,) = _forma_lines(caplog)
    assert re.search(r"secure_pages=3 cap=3 over_cap_nodes=\d+ kept_worker_evicted=\d+ kept_unkeyed=\d+ ms=\d+\.\d", line), line


def test_a3_default_cap_is_128_and_the_small_spill_is_unchanged(group, monkeypatch, caplog):
    from sglang.srt.environ import envs

    g = group
    monkeypatch.delenv(CAP_ENV, raising=False)
    assert envs.SGLANG_WEG2_W3_FORMA_SPILL_MAX_PAGES.get() == 128
    caplog.set_level(logging.INFO)
    _tp0_serves_a_worker_request(g)
    assert all(n.component_data[FULL].host_value is None for n in g.n0), "8 pages < 128: every node spilled"
    (line,) = _forma_lines(caplog)
    assert "cap=128 over_cap_nodes=0" in line and " ms=" in line


# ------------------------------------------------------------------------------------------------ A4

def _follower(rid, told, live):
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1, pp_size=3),
                                  tree_cache=types.SimpleNamespace())
    sched._weg2_fb_follower = FB._FState()            # the PF ack stream exists: the hold re-asks, never stops
    sched._w27u_unreached = {rid: (told, live)}
    return sched, types.SimpleNamespace(rid=rid)


def _tripwire(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("PF TOLD-FLOOR-THEN-W27")]


def test_a4_a_w27_held_after_the_floor_of_the_same_rid_is_named_once(caplog):
    caplog.set_level(logging.WARNING)
    sched, req = _follower("weg2-3-27", 87296, 65536)
    FB.apply_floor(sched, "weg2-3-27", 87296)                   # the floor marker on this follower
    assert ST.follower_hold_unreached(sched, req, "admission") is True
    assert ST.follower_hold_unreached(sched, req, "admission") is True   # a held rid visits again: named once
    lines = _tripwire(caplog)
    assert len(lines) == 1 and "rid=weg2-3-27 floor=87296 w27=HELD" in lines[0], lines


def test_a4_no_floor_or_outside_the_window_names_nothing(caplog):
    caplog.set_level(logging.WARNING)
    sched, req = _follower("weg2-4-25", 43392, 26880)
    assert ST.follower_hold_unreached(sched, req, "admission") is True   # a hold without a floor: W27's own line
    assert _tripwire(caplog) == []
    FB.note_floor_applied(sched, "weg2-4-25", 43392, now=1000.0)
    assert FB.floor_then_w27(sched, "weg2-4-25", "SHORT", 43392, 26880, now=1000.0 + FB.FLOOR_W27_WINDOW_S + 1) is False
    assert FB.floor_then_w27(sched, "weg2-4-25", "SHORT", 43392, 26880, now=1010.0) is True
    assert FB.floor_then_w27(sched, "weg2-4-25", "SHORT", 43392, 26880, now=1011.0) is False   # once per kind
    assert FB.floor_then_w27(sched, "weg2-4-25", "HELD", 43392, 26880, now=1012.0) is True


def test_a4_the_short_site_asks_the_tripwire():
    """The SHORT line sits deep in follower_reach_told (rank_resumable, re-read): its call is pinned by the call it
    makes before the credit pop -- a source anchor, the HELD half above is the behaviour."""
    import inspect

    src = inspect.getsource(ST.follower_reach_told)
    i_line = src.index("W27-UNIFORM FOLLOWER TREE SHORT")
    i_wire = src.index('_fb.floor_then_w27(scheduler, rid, "SHORT"')
    assert i_line < i_wire < src.index("_pop_credit_keep_pin(tree, rid)", i_wire)


def test_a4_the_tripwire_never_raises():
    assert FB.floor_then_w27(object(), "x", "SHORT", 1, 0) is False
    FB.note_floor_applied(object(), "x", 1)      # no attribute can be set: silently nothing
