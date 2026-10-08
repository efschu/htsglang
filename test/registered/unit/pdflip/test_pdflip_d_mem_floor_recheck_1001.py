"""NF y6k 01.10. (boot ...182252Z-db1d, D-TP0 py-spy 15 s): 31.9 % of the
D-TP0 scheduler sat in ``max_live_page`` -- get_next_batch_to_run ->
d_seat_rewake.round_boundary -> d_seat_vram.runtime_tick ->
_group_floor_tokens. Under a pending shrink (the retained tree holds pages
above the wanted stage) the tick re-read the group floor EVERY decode round:
the free lists copied device->host (a sync of the decode stream) plus one
group collective, and the room-below-the-cap verdict the same way. Steady
decode round on D-TP0: 37 ms wall / 28 ms GPU (host gap 9 ms), x176 before
D-MEM-SCHED: 24 / 21 / 2.9.

WHAT MUST HOLD. A steady decode round under an unchanged pending shrink
reads neither the floor nor the room every round; an end event still shrinks
at once (the END shrink: KV is freed at finish), a drained floor is taken
within the re-check bound, and every rank enters the group collective in the
same rounds (replicated inputs only).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER, _req, floor_env, tick_env  # noqa: E402,F401

ROUNDS = 256


def _alloc(free_ids):
    return types.SimpleNamespace(free_pages=torch.tensor(free_ids, dtype=torch.int64),
                                 release_pages=torch.empty(0, dtype=torch.int64))


def _count_reads(monkeypatch, dsv, floor):
    reads = {"floor": 0, "room": 0}
    monkeypatch.setattr(dsv, "max_live_page",
                        lambda alloc: reads.__setitem__("floor", reads["floor"] + 1) or floor["page"])
    real = dsv.free_tokens_below

    def _room(*a, **k):
        reads["room"] += 1
        return real(*a, **k)

    monkeypatch.setattr(dsv, "free_tokens_below", _room)
    return reads


def _pending_with_a_decoding_seat(dsv, sched, caps, floor):
    """pdflip-2-10 finished at S1; its retained pages (517) hold the shrink to
    S0 pending; a small seat keeps decoding below the pending cap."""
    sched.running_batch.reqs = [_req("pdflip-2-10", 32835, 258)]
    dsv.runtime_tick(sched)
    sched.running_batch.reqs = []
    floor["page"] = 517
    dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert ms.pending == 0 and caps[-1] == FLOOR_LADDER[0]
    sched.running_batch.reqs = [_req("seat", 1000, 10)]
    dsv.runtime_tick(sched)
    assert ms.pending == 0 and not getattr(ms, "_cap_lifted", False)
    return ms


def test_a_steady_pending_round_reads_no_floor_and_no_room(floor_env, monkeypatch):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))   # room below the cap
    reads = _count_reads(monkeypatch, dsv, floor)
    ms = _pending_with_a_decoding_seat(dsv, sched, caps, floor)
    f0, r0 = reads["floor"], reads["room"]
    for _ in range(ROUNDS):
        st = dsv.runtime_tick(sched)
        assert not st.changed and ms.pending == 0
    floor_reads, room_reads = reads["floor"] - f0, reads["room"] - r0
    # the old tick read both every round (ROUNDS device->host syncs + 2*ROUNDS collectives)
    assert floor_reads <= 12, "floor re-read %d times in %d steady rounds" % (floor_reads, ROUNDS)
    assert room_reads <= 12, "room re-read %d times in %d steady rounds" % (room_reads, ROUNDS)
    assert caps[-1] == FLOOR_LADDER[0]                  # the pending cap stays engaged


def test_an_end_event_still_shrinks_at_once(floor_env, monkeypatch):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))
    _count_reads(monkeypatch, dsv, floor)
    ms = _pending_with_a_decoding_seat(dsv, sched, caps, floor)
    for _ in range(40):                                 # deep into the back-off
        dsv.runtime_tick(sched)
    floor["page"] = 0                                   # the retained tree is gone ...
    sched.running_batch.reqs = []                       # ... and the seat finished
    st = dsv.runtime_tick(sched)
    assert st.changed and ms.pending is None and ms.stage == 0, (
        "an end event must re-read the floor at once (END shrink), got stage S%d pending %s"
        % (ms.stage, ms.pending))


def test_a_drained_floor_is_taken_within_the_recheck_bound(floor_env, monkeypatch):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))
    _count_reads(monkeypatch, dsv, floor)
    ms = _pending_with_a_decoding_seat(dsv, sched, caps, floor)
    for _ in range(100):
        dsv.runtime_tick(sched)
    floor["page"] = 0                                   # evicted without an end event
    k = dsv._recheck_max()
    for i in range(k + 1):
        st = dsv.runtime_tick(sched)
        if st.changed:
            break
    assert ms.pending is None and ms.stage == 0, "the drained floor waited more than %d rounds" % k


def test_recheck_one_restores_the_per_round_read(floor_env, monkeypatch):
    dsv, sched, caps, floor = floor_env
    monkeypatch.setenv("FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS", "1")
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))
    reads = _count_reads(monkeypatch, dsv, floor)
    _pending_with_a_decoding_seat(dsv, sched, caps, floor)
    f0 = reads["floor"]
    for _ in range(20):
        dsv.runtime_tick(sched)
    assert reads["floor"] - f0 == 20


def test_a_lift_never_shrinks_on_a_stale_floor(floor_env, monkeypatch):
    """While the cap is lifted to the stage, pages may go above the old floor:
    a cached round must not shrink below them."""
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(600, 1024)))  # no room below the cap
    _count_reads(monkeypatch, dsv, floor)
    sched.running_batch.reqs = [_req("pdflip-2-10", 32835, 258)]
    dsv.runtime_tick(sched)
    sched.running_batch.reqs = []
    floor["page"] = 517
    dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    sched.running_batch.reqs = [_req("seat", 1000, 10)]
    dsv.runtime_tick(sched)
    assert getattr(ms, "_cap_lifted", False) and caps[-1] == FLOOR_LADDER[1]
    for _ in range(30):
        st = dsv.runtime_tick(sched)
        assert not st.changed and ms.stage == 1


def test_every_rank_enters_the_collective_in_the_same_rounds(tick_env, monkeypatch):
    """Two ranks with different free lists (uneven DCP KV) must call the group
    MIN in the same ticks -- the re-check is driven by replicated inputs and
    the collective's own (replicated) answer only."""
    from flliper.srt.pdflip import d_seat_vram as dsv

    dsv_, _sched, _caps, _floor, _votes, _grid = tick_env
    monkeypatch.setenv("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", ",".join(str(t) for t in FLOOR_LADDER))
    floor = {"page": 0}
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: floor["page"])
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: t)
    calls = {0: [], 1: []}
    peer = {"slack": 5000}

    def mk(rank, free_ids):
        tick_no = {"n": 0}

        def gmin(vals):
            calls[rank].append(tick_no["n"])
            vals = list(vals)
            if len(vals) == 2:          # [ok, slack]: the group answers its MIN slack
                return [1, peer["slack"]]
            return vals

        s = types.SimpleNamespace(
            server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                              speculative_num_draft_tokens=4),
            running_batch=types.SimpleNamespace(reqs=[]), waiting_queue=[], chunked_req=None,
            page_size=64, token_to_kv_pool_allocator=_alloc(free_ids), _pdflip_group_min_ints=gmin)
        setattr(s, dsv.CTL_ATTR, False)
        setattr(s, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e4", n=1, cap=6, done=True, stage=1,
                                                  stage_tokens=FLOOR_LADDER[1]))
        return s, tick_no

    ranks = [mk(0, list(range(100, 512))), mk(1, list(range(300, 512)))]

    def tick_all(reqs):
        for s, tn in ranks:
            tn["n"] += 1
            s.running_batch.reqs = list(reqs)
            dsv.runtime_tick(s)

    tick_all([_req("pdflip-2-10", 32835, 258)])
    floor["page"] = 517
    tick_all([])
    for _ in range(120):
        tick_all([_req("seat", 1000, 10)])
    assert calls[0] == calls[1] and len(calls[0]) < 40
