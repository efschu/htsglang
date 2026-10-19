"""S72 -- a HELD demand does not re-read the room below the pending cap in every pass.

MEASURED (rc16 boot ..._bf9ca46f45_1019_121629.D.log, TP0, 13:28:43-13:30:58, bs 1, one queued request
pdflip-137-471 with a 144896-token load-back, DQH holding the head): ``PDFLIP D-MEM-SCHED FLOOR-CHECK``
``room_reads`` +512 per 512 ticks and ``room_cached=0`` with ``pending=S8 lifted=yes`` (7 of 7 such windows),
``DECODE-HOST-SPLIT sched_ms`` 31.5-45.2 (3.7 in the 101 windows without the reads), ``sync_ms`` 0 (the
overlap gone), ``Decode rank batch`` 42 ms wall / 33.5 gpu-ms for the 3157 bs1 rounds of pdflip-132-465 --
the whole rc16 "bs1 64-160k" bucket. Cause: ``_room_ok`` bypassed its cache for ANY ``incoming > 0``; each
read is a device->host copy of the free lists (behind the schedule stream's WAR wait on the previous
forward) plus one group collective.

WHAT MUST HOLD (driven through the REAL ``runtime_tick`` on 3 rank doubles meeting in one rendezvous,
the harness of test_pdflip_cand3_rank_stepping_1536):
  h1  an unchanged held demand under a lift reads the room on the back-off (1, 2, 4 .. K), not every pass;
  h2  the hit test reads only replicated inputs: rank-local load-back extents (``need`` differs per rank,
      and changes on ONE rank mid-hold) never split the collective count;
  h3  a changed demand (another queued rid, an admitted head) re-reads at once;
  h4  an OK verdict under demand is still read every pass (NF1d: an admission must not ride a stale
      "the pending cap pays it");
  h5  the lift ends on a fresh verdict within K passes once the room is there;
  h6  without a demand fingerprint (direct callers) the old rule stands.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import types  # noqa: E402

from test_pdflip_cand3_rank_stepping_1536 import (  # noqa: E402,F401
    N,
    _alloc,
    _decode_only_cache,
    _req,
    _room_stats,
    _state,
    _tick,
    floor_env,
    group,
    tick_env,
)
from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER  # noqa: E402

K = 64
TICKS = 256


def _held_head(dsv, scheds, rv, monkeypatch, extents=(22000, 0, 8000), rid="pdflip-137-471"):
    """Pending S0 (cap 32768), one decoding seat, room 22400 tokens below the cap on every rank; a queued
    head of 24000 tokens whose host load-back extent was stamped at each rank's own match (#1046)."""
    _decode_only_cache(dsv, scheds, rv, monkeypatch, k=K)
    for i, s in enumerate(scheds):
        s.token_to_kv_pool_allocator = _alloc(list(range(10, 360)) + list(range(600, 700)))
        r = _req(rid, 24000)
        r.pp_load_back_extent = extents[i]
        s.waiting_queue = [r]
    rv.reset()


def _run(dsv, scheds, rv, ticks):
    """``ticks`` lockstep ticks; per tick: no error, the same collective count on every rank, the same
    fresh room reads on every rank. Returns the fresh room reads per tick (rank 0)."""
    per_tick = []
    for _ in range(ticks):
        rv.reset()
        before = _room_stats(scheds, dsv)
        errs = _tick(dsv, scheds)
        assert errs == [None] * N, [repr(e) for e in errs]
        assert len(set(rv.count())) == 1, rv.count()
        reads = {a[0] - b[0] for a, b in zip(_room_stats(scheds, dsv), before)}
        assert len(reads) == 1, reads
        per_tick.append(reads.pop())
    return per_tick


def test_h1_a_held_demand_under_a_lift_reads_on_the_back_off(group, monkeypatch):
    dsv, scheds, rv = group
    _held_head(dsv, scheds, rv, monkeypatch)
    reads = _run(dsv, scheds, rv, TICKS)
    assert _state(dsv, scheds) == [(True, FLOOR_LADDER[1])] * N        # the lift stands
    # old rule: 256 fresh reads (one device->host copy + one collective per pass)
    assert sum(reads) <= 12, "held demand re-read the room %d times in %d passes" % (sum(reads), TICKS)
    assert reads[0] == 1                                                 # the first pass reads


def test_h2_rank_local_extents_changing_on_one_rank_keep_the_collective_count_uniform(group, monkeypatch):
    dsv, scheds, rv = group
    _held_head(dsv, scheds, rv, monkeypatch)
    _run(dsv, scheds, rv, 40)
    scheds[1].waiting_queue[0].pp_load_back_extent = 23000   # TP1 re-matched: its need grows alone
    _run(dsv, scheds, rv, 80)
    scheds[0].waiting_queue[0].pp_load_back_extent = 0       # TP0's drops to nothing
    _run(dsv, scheds, rv, 80)
    assert len({st for st in _state(dsv, scheds)}) == 1


def test_h3_a_changed_demand_reads_at_once(group, monkeypatch):
    dsv, scheds, rv = group
    _held_head(dsv, scheds, rv, monkeypatch)
    _run(dsv, scheds, rv, 40)                                 # deep in the back-off
    for s in scheds:                                          # a second request queues behind the head
        s.waiting_queue = s.waiting_queue + [_req("pdflip-138-472", 2000)]
    assert _run(dsv, scheds, rv, 1) == [1]
    _run(dsv, scheds, rv, 40)
    for s in scheds:                                          # another rid of the SAME size replaces it
        s.waiting_queue = [s.waiting_queue[0], _req("pdflip-139-480", 2000)]
    assert _run(dsv, scheds, rv, 1) == [1]
    _run(dsv, scheds, rv, 40)
    for s in scheds:                                          # the head is admitted: running set changes
        s.running_batch.reqs = s.running_batch.reqs + [s.waiting_queue[0]]
        s.waiting_queue = s.waiting_queue[1:]
    assert _run(dsv, scheds, rv, 1) == [1]


def test_h4_an_ok_verdict_under_demand_is_still_read_every_pass(group, monkeypatch):
    dsv, scheds, rv = group
    _held_head(dsv, scheds, rv, monkeypatch)
    for s in scheds:                                          # the room below the cap pays the head
        s.token_to_kv_pool_allocator = _alloc(list(range(10, 500)))
        s.token_to_kv_pool_allocator.caps = [FLOOR_LADDER[0]]       # the pending cap is engaged
        s.waiting_queue = [_req("pdflip-1-2", 2000)]
    reads = _run(dsv, scheds, rv, 16)
    assert _state(dsv, scheds) == [(False, FLOOR_LADDER[0])] * N       # pending cap, no lift
    assert reads == [1] * 16, reads


def test_h5_the_lift_ends_on_a_fresh_verdict_within_k(group, monkeypatch):
    dsv, scheds, rv = group
    _held_head(dsv, scheds, rv, monkeypatch)
    _run(dsv, scheds, rv, 100)                                # gap at K
    for s in scheds:                                          # room appears below the cap, demand unchanged
        s.token_to_kv_pool_allocator = _alloc(list(range(10, 512)))
        s.token_to_kv_pool_allocator.caps = [FLOOR_LADDER[1]]
    for i in range(K + 1):
        _run(dsv, scheds, rv, 1)
        if not any(st[0] for st in _state(dsv, scheds)):
            break
    assert _state(dsv, scheds) == [(False, FLOOR_LADDER[0])] * N, "the lift outlived %d passes" % K


def test_h6_without_a_demand_fingerprint_the_old_rule_stands(monkeypatch):
    from flliper.srt.pdflip import d_seat_vram as V

    monkeypatch.setenv("FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS", "8")
    seen = []
    sched = types.SimpleNamespace(_pdflip_group_min_ints=lambda v: seen.append(list(v)) or list(v),
                                  server_args=None)
    alloc = _alloc([1, 2])                                    # room 128 < need 4096: NOT ok
    ms = types.SimpleNamespace(pending=1, _round=0)
    fr = V._floor_room_state(ms)
    for _ in range(4):
        assert V._room_ok(sched, ms, fr, alloc, 4096, 4096, 128, 1, 64) is False
    assert len(seen) == 4
    for _ in range(4):
        assert V._room_ok(sched, ms, fr, alloc, 4096, 4096, 128, 1, 64, demand=("r1",)) is False
    assert len(seen) == 5                                     # one read, then the held lift is reused
