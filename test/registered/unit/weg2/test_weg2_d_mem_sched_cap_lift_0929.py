"""NF1d 09291811 (z30y3f @ e952a6b709): D died in the first bs2 cell.

THE METAL. bs1 2k..245k ran; the 245k request ended at S7 and the end event
left a pending shrink to S0 (18:21:35 ``cap=32768 pending S-->S0 stage=S7
floor=248832``: the retained tree holds pages up to 248832). The first bs2
cell's 2113 tokens fit S0, so nothing cancelled the pending shrink and the
cap stayed at 32768. Every id below it was the tree's, every leaf the peel
freed sat above it: "EVICTION UNDER-DELIVERED: asked for 2177 tokens, the
pool received 0 ... A RESIDENCY CAP IS ENGAGED and is holding 278208 slot
ids" -> ``RuntimeError: Prefill out of memory`` on TP0/TP1/TP2 at once, the
abort's GPU coredumps, W17_Weg2GroupDead.

WHAT MUST HOLD. While a shrink is pending, the group's demand is paid: if the
free ids below the pending cap cannot pay it on every rank (replicated MIN),
the cap goes back to the MAPPED stage; with no demand left it returns to the
pending cap. Room below the cap keeps the design (new pages below).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import types  # noqa: E402

import torch  # noqa: E402

from test_weg2_d_mem_sched_0929 import FLOOR_LADDER, _req, floor_env, tick_env  # noqa: E402,F401


def _alloc(free_ids):
    return types.SimpleNamespace(free_pages=torch.tensor(free_ids, dtype=torch.int64),
                                 release_pages=torch.empty(0, dtype=torch.int64))


def _pending_s0(dsv, sched, caps, floor):
    """weg2-2-10 at S1 finishes; its retained pages hold the shrink: pending S0."""
    sched.running_batch.reqs = [_req("weg2-2-10", 32835, 258)]
    dsv.runtime_tick(sched)
    sched.running_batch.reqs = []
    floor["page"] = 517
    st = dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert not st.changed and ms.pending == 0 and caps[-1] == FLOOR_LADDER[0]
    return ms


def test_a_pending_cap_without_room_below_hands_the_admission_the_mapped_stage(floor_env):
    dsv, sched, caps, floor = floor_env
    # page 64: the cap of 32768 is page 512; the tree holds every id below it
    sched.token_to_kv_pool_allocator = _alloc(list(range(600, 1024)))
    ms = _pending_s0(dsv, sched, caps, floor)
    sched.waiting_queue = [_req("bs2-a", 2113)]          # fits S0: cancels nothing
    st = dsv.runtime_tick(sched)
    assert not st.changed and ms.pending == 0 and ms.stage == 1
    assert caps[-1] == FLOOR_LADDER[1], (
        "the pending cap %d stayed engaged with no free id below it: the 2113-token "
        "admission cannot be paid (NF1d Prefill out of memory)" % caps[-1])
    # the request runs: still no room below -> the stage keeps paying
    sched.waiting_queue = []
    sched.running_batch.reqs = [_req("bs2-a", 2113, 10)]
    n = len(caps)
    dsv.runtime_tick(sched)
    assert len(caps) == n and caps[-1] == FLOOR_LADDER[1]
    # the group is idle again: back to the pending cap
    sched.running_batch.reqs = []
    dsv.runtime_tick(sched)
    assert caps[-1] == FLOOR_LADDER[0] and ms.pending == 0


def test_room_below_the_pending_cap_keeps_it(floor_env):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))  # 412 pages below
    ms = _pending_s0(dsv, sched, caps, floor)
    n = len(caps)
    sched.waiting_queue = [_req("bs2-a", 2113)]
    dsv.runtime_tick(sched)
    assert len(caps) == n and caps[-1] == FLOOR_LADDER[0] and ms.pending == 0


def test_one_rank_without_room_lifts_the_cap_on_every_rank(floor_env):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))  # room HERE
    sched._weg2_group_min_ints = lambda vals: [min(list(vals) + [0])]  # a peer has none
    _pending_s0(dsv, sched, caps, floor)
    sched.waiting_queue = [_req("bs2-a", 2113)]
    dsv.runtime_tick(sched)
    assert caps[-1] == FLOOR_LADDER[1]


def test_the_lift_writes_its_line(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(600, 1024)))
    _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    sched.waiting_queue = [_req("bs2-a", 2113)]
    dsv.runtime_tick(sched)
    lines = [m for m in caplog.messages if dsv.CAP_LIFT_MARK in m]
    assert lines and "lifted=yes" in lines[-1] and "cap=%d" % FLOOR_LADDER[1] in lines[-1]


def test_free_tokens_below_counts_only_ids_under_the_cap():
    from sglang.srt.weg2 import d_seat_vram as dsv

    a = _alloc([1, 2, 511, 512, 513, 900])
    assert dsv.free_tokens_below(a, 32768, 64) == 4 * 64
    assert dsv.free_tokens_below(object(), 32768, 64) is None
