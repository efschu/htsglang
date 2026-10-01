# SPDX-License-Identifier: Apache-2.0
"""L15-11b: the hold never lands on padding slot 0; mamba allocator re-armed.

Global KV slot 0 and mamba slot 0 are the DUMMY WRITE TARGET for padded
tokens: both allocator clear() sites hand out ``arange(1, size + 1)``, so
slot 0 is never free and every padded token WRITES into it. A held slot or
a held anchor compacted into slot 0 therefore either crashes in
``reserve_slots`` at step 6 (after the buffers moved) or silently corrupts
the hold. This suite pins the fix: ``compact_plan``/``anchor_plan`` take a
``reserved`` set (default ``()`` = old behaviour), ``retain_at_sleep``
passes ``PAD_SLOTS`` and re-arms the mamba allocator via
``reserve_mamba_slots``. Plain pytest functions on purpose.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.weg2 import l15_retain
from sglang.srt.weg2.l15_compact import anchor_plan, compact_plan

from weg2.test_weg2_l15_retain_0930 import (  # noqa: E402 - module-level fakes
    ALLOC_SIZE,
    CANDIDATES,
    CAPS_ROWS_BY_RANK,
    CAP_ANCHOR_SLOTS,
    FakeAllocator,
    FakeNode,
    _marker,
)

# Two ranks, S=4: rank 0 owns residues {0,1}, rank 1 owns residues {2,3}.
PREFIX2 = (0, 2, 4)


class FakeMambaAllocator:
    """Mirrors MambaSlotAllocator.clear(): free_slots = arange(1, size + 1),
    int64, slot 0 reserved for padded tokens and never handed out."""

    def __init__(self, size=16):
        self.size = size
        self.free_slots = torch.empty(0, dtype=torch.int64)

    def clear(self):
        self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int64)


def test_compact_plan_reserved_zero_is_never_a_target():
    # held 5 (class r0) and 6 (class r1), L_H=4: the natural free targets of
    # class r0 inside [0,4) are [0,1]; slot 0 must be skipped.
    held = {"h1": (5, 6)}
    plan = compact_plan(held, PREFIX2, reserved=(0,))
    assert plan.l_h == 4  # L_H covers the reserved slot: 0 < l_h
    assert plan.moves == ((5, 1), (6, 2))
    assert plan.new_slots == {"h1": (1, 2)}
    assert 0 not in plan.new_slots["h1"]
    assert all(new != 0 for _, new in plan.moves)


def test_compact_plan_default_reserved_unchanged():
    # reserved=() keeps today's behaviour: without the fix the class-r0
    # source 5 would compact onto padding slot 0 (the bug this file pins).
    plan = compact_plan({"h1": (5, 6)}, PREFIX2)
    assert plan.moves == ((5, 0), (6, 2))
    assert plan.new_slots == {"h1": (0, 2)}


def test_anchor_plan_reserved_zero_excluded_and_anchor_at_zero_rejected():
    a_h, moves = anchor_plan({"a": 5, "b": 6, "c": 7}, reserved=(0,))
    assert a_h == 4  # 3 anchors + 1 reserved slot
    assert moves == ((5, 1), (6, 2), (7, 3))
    assert all(new != 0 for _, new in moves)
    with pytest.raises(ValueError, match="reserved"):
        anchor_plan({"a": 0}, reserved=(0,))
    # default: exactly the old squeeze into [0, A_H)
    a_h0, moves0 = anchor_plan({"a": 5, "b": 6, "c": 7})
    assert (a_h0, moves0) == (3, ((5, 0), (6, 1), (7, 2)))


def test_reserve_mamba_slots_refuses_a_slot_that_is_not_free():
    alloc = FakeMambaAllocator()
    alloc.clear()
    assert l15_retain.reserve_mamba_slots(alloc, [5]) == 1
    assert 5 not in alloc.free_slots.tolist()
    with pytest.raises(ValueError, match="not free"):
        l15_retain.reserve_mamba_slots(alloc, [5])  # already taken
    with pytest.raises(ValueError, match="not free"):
        l15_retain.reserve_mamba_slots(alloc, [0])  # padded slot 0


def test_retain_at_sleep_end_to_end_avoids_slot0_and_rearms_mamba(tmp_path):
    # Reuses the proven candidate set of the 0930 suite (select_hold keeps
    # r_seat + r_parked). Geometry hits slot 0 WITHOUT the reserved rule:
    # r_seat (5,6) r_parked (13,) under PREFIX2 -> r0 need 3 (incl. the
    # reserved 0) -> L_H=8, 13 -> 1; without reserved: 5 -> 0 (crash).
    events = []
    nodes = {"r_seat": FakeNode(events), "r_parked": FakeNode(events)}
    alloc = FakeAllocator(events)
    mamba_alloc = FakeMambaAllocator()
    set_keep_calls = []

    def set_keep(ptr, ranges):
        events.append("set_keep")
        set_keep_calls.append(("set_keep", id(ptr), tuple(ranges)))

    kv_buf = _marker(6, 3)    # rows_by_rank = (4, 4) <= 6
    mamba_buf = _marker(16, 2)  # anchor sources 4, 9 need >= 10 rows

    res = l15_retain.retain_at_sleep(
        candidates=list(CANDIDATES),
        node_of=lambda rid: nodes[rid],
        slots_of=lambda rid: {"r_seat": (5, 6), "r_parked": (13,)}[rid],
        anchor_slot_of=lambda rid: {"r_seat": 4, "r_parked": 9}[rid],
        l2_of=lambda rid: ((201, 202), (5, 6)),
        caps_rows_by_rank=CAPS_ROWS_BY_RANK,
        cap_anchor_slots=CAP_ANCHOR_SLOTS,
        prefix=PREFIX2,
        rank=1,
        epoch=91,
        pid=4242,
        kv_buffers=[kv_buf],
        mamba_buffers=[mamba_buf],
        allocator=alloc,
        mamba_allocator=mamba_alloc,
        reset_keep=lambda hold_nodes: events.append("reset_keep"),
        set_keep=set_keep,
        manifest_path=str(tmp_path / "l15_manifest.json"),
        log=lambda line: None,
    )
    assert res is not None
    # slot 0 nowhere in the KV plan
    for slots in res.plan.new_slots.values():
        assert 0 not in slots
    assert all(new != 0 for _, new in res.plan.moves)
    assert res.plan.l_h == 8 and res.plan.rows_by_rank == (4, 4)
    # KV allocator: 0 was never free and the held new slots left free_pages
    free = alloc.free_pages.tolist()
    assert 0 not in free
    assert set(res.plan.new_slots["r_seat"]) | set(res.plan.new_slots["r_parked"]) <= {
        s for s in range(1, ALLOC_SIZE + 1) if s not in free
    }
    # anchors squeezed to [0, A_H) around the reserved slot; A_H covers it
    assert res.a_h == 3
    assert nodes["r_seat"].anchor_slot == 1
    assert nodes["r_parked"].anchor_slot == 2
    # mamba re-arm: 0 never in free_slots, held anchors removed, rest stays
    mfree = mamba_alloc.free_slots.tolist()
    assert mfree == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # keep windows: kv [0, rows), mamba [0, A_H) -- slot 0 stays mapped
    kv_calls = [c for c in set_keep_calls if c[1] == id(kv_buf)]
    mb_calls = [c for c in set_keep_calls if c[1] == id(mamba_buf)]
    assert kv_calls == [("set_keep", id(kv_buf), ((0, 4),))]
    assert mb_calls == [("set_keep", id(mamba_buf), ((0, 3),))]
