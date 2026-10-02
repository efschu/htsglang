# SPDX-License-Identifier: Apache-2.0
"""L15-12c-F7: shared-prefix requests must not kill the retain round.

Under the radix cache, concurrent requests sharing a prefix SHARE KV slots
(and, ending on the same node, the same mamba anchor). The compaction map is
a function of the SLOT, so a slot held by several rids is planned ONCE:
counted once in need/capacity, given one new slot, and every rid that holds
it maps through the same kv_map/anchor entry. A slot repeated INSIDE one
rid's own held list stays an error (corruption, not sharing).

Before this fix, compact_plan raised "slot held twice" and anchor_plan
raised "anchor slot held by both", so the whole retain round was skipped
under agent load (one common system prompt) and L1.5 never retained.

Geometry (test_weg2_l15_retain_0930 style): prefix (0, 1, 2) -> S = 2,
rank 0 owns even global slots, rank 1 owns odd.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.weg2 import l15_compact, l15_retain
from sglang.srt.weg2.l15_policy import Candidate, HoldSet

PREFIX = (0, 1, 2)


def test_compact_plan_shared_prefix_counts_slot_once():
    # sa and sb share slot 10 (their first slot); sa holds 12, sb holds 14.
    # All three are even -> rank 0. Shared counted once => need=[3, 0].
    plan = l15_compact.compact_plan(
        {"sa": (10, 12), "sb": (10, 14)}, PREFIX
    )
    # Shared slot 10 counted once in need: need=[3, 0] -> L_H = 6 (not 8).
    assert plan.l_h == 6
    # The shared slot gets ONE new slot (10 -> 0), and both rid maps agree.
    assert (10, 0) in plan.moves
    assert plan.new_slots["sa"][0] == plan.new_slots["sb"][0] == 0
    # Each rid's own unique slot still lands on its own new row.
    assert plan.new_slots["sa"] == (0, 2)
    assert plan.new_slots["sb"] == (0, 4)


def test_compact_plan_shared_prefix_still_counts_need_by_owner():
    # Shared slots on BOTH ranks: sa/sb share (2, 3); sa adds (4,), sb adds (5,).
    # 2 even (rank0), 3 odd (rank1), 4 even (rank0), 5 odd (rank1).
    plan = l15_compact.compact_plan(
        {"sa": (2, 3, 4), "sb": (2, 3, 5)}, PREFIX
    )
    # need: rank0 {2, 4} -> 2, rank1 {3, 5} -> 2. L_H = 4.
    assert plan.l_h == 4
    # Shared slots stay put (< L_H) and both maps agree on them.
    assert plan.new_slots["sa"][0] == plan.new_slots["sb"][0] == 2
    assert plan.new_slots["sa"][1] == plan.new_slots["sb"][1] == 3
    # 4 and 5 are >= L_H -> they move to the free rows 0 and 1 of their
    # own class; shared 2 and 3 stay.
    assert plan.new_slots["sa"] == (2, 3, 0)
    assert plan.new_slots["sb"] == (2, 3, 1)


def test_compact_plan_intra_rid_duplicate_still_raises():
    # The SAME slot twice inside ONE rid's list is corruption, not sharing.
    with pytest.raises(ValueError):
        l15_compact.compact_plan({"sa": (10, 12, 10)}, PREFIX)


def test_compact_plan_two_rids_never_share_within_one_rid():
    # A slot shared across rids is fine, but a rid listing it twice is not.
    with pytest.raises(ValueError):
        l15_compact.compact_plan({"sa": (10, 10), "sb": (12,)}, PREFIX)


def test_anchor_plan_shared_anchor_planned_once():
    # Both rids end on the same node -> the SAME mamba anchor slot 11.
    a_h, moves = l15_compact.anchor_plan(
        {"sa": 11, "sb": 11}, reserved=(0,)
    )
    # One unique anchor (11) + reserved pad (0) -> A_H = 2; 11 squeezes to 1.
    assert a_h == 2
    assert moves == ((11, 1),)


def test_anchor_plan_disjoint_anchors_unaffected():
    # Regression guard: distinct anchors still squeeze the usual way.
    a_h, moves = l15_compact.anchor_plan({"sa": 9, "sb": 11}, reserved=(0,))
    assert a_h == 3
    assert moves == ((9, 1), (11, 2))


def _mamba_fake(size=16):
    class FakeMambaAllocator:
        def __init__(self):
            self.size = size
            self.free_slots = torch.empty(0, dtype=torch.int64)

        def clear(self):
            self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int64)

    return FakeMambaAllocator()


def _kv_marker(rows):
    return torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)


def test_retain_at_sleep_shared_prefix_is_retained(tmp_path, monkeypatch):
    # Two served requests sharing their first two slots (10, 11) and the
    # SAME mamba anchor (12). select_hold is pinned to keep exactly these two.
    hs = HoldSet(rids=("sa", "sb"), rows_by_rank=(2, 2), anchors=2, excluded=())
    monkeypatch.setattr(l15_retain, "select_hold", lambda *a, **k: hs)

    kv_buf = _kv_marker(16)
    mamba_buf = torch.arange(16 * 2, dtype=torch.float32).reshape(16, 2)

    class FakeAllocator:
        size = 16
        free_pages = torch.empty(0, dtype=torch.int64)
        release_pages = torch.empty(0, dtype=torch.int64)

        def clear(self):
            self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
            self.release_pages = torch.empty(0, dtype=torch.int64)

    class FakeNode:
        pass

    alloc = FakeAllocator()
    mamb = _mamba_fake()
    set_keep_calls = []

    candidates = [
        Candidate(rid="sa", kind="served", last_active=9.0,
                  rows_by_rank=(1, 1), anchor_depth=2, kv_depth=2),
        Candidate(rid="sb", kind="served", last_active=8.0,
                  rows_by_rank=(1, 1), anchor_depth=2, kv_depth=2),
    ]
    # Shared first two slots (10 even, 11 odd), unique tail (12 even, 13 odd).
    slots_of = {"sa": (10, 11, 12), "sb": (10, 11, 13)}
    anchor_of = {"sa": 12, "sb": 12}

    res = l15_retain.retain_at_sleep(
        candidates=candidates,
        node_of=lambda rid: FakeNode(),
        slots_of=lambda rid: slots_of[rid],
        anchor_slot_of=lambda rid: anchor_of[rid],
        l2_of=lambda rid: ((201, 202), (5, 6)),
        rewrite_tree=lambda node, kv_map, anchor_map, visited: None,
        caps_rows_by_rank=(4, 4),
        cap_anchor_slots=4,
        prefix=PREFIX,
        rank=1,
        epoch=77,
        pid=4242,
        kv_buffers=[kv_buf],
        mamba_buffers=[mamba_buf],
        allocator=alloc,
        mamba_allocator=mamb,
        reset_keep=lambda nodes: None,
        set_keep=lambda ptr, ranges: set_keep_calls.append(ranges),
        manifest_path=str(tmp_path / "l15_manifest.json"),
        log=lambda line: None,
    )
    # Before the fix compact_plan/anchor_plan raise -> the round is skipped
    # (res is None). After the fix the shared pair is retained.
    assert res is not None
    # The manifest carries the NEW (remapped) slots. With the reserved
    # padding slot 0 the need is r0 = 3 (10, 12, pad) / r1 = 2 (11, 13)
    # -> L_H = 6, and the free class rows give 10 -> 2, 11 -> 1, 12 -> 4,
    # 13 -> 3. Both spans carry the shared images (2, 1) at the same
    # indices, each slot exactly once per span.
    by_rid = {span.rid: span for span in res.manifest.spans}
    assert by_rid["sa"].slots == (2, 1, 4)
    assert by_rid["sb"].slots == (2, 1, 3)
    assert by_rid["sa"].slots[:2] == by_rid["sb"].slots[:2]
    assert len(set(by_rid["sa"].slots)) == len(set(by_rid["sb"].slots)) == 3
    # The shared anchor lands on ONE slot and both spans agree on it.
    assert by_rid["sa"].anchor_slot == by_rid["sb"].anchor_slot
    # The keep window is a single contiguous range per buffer.
    assert set_keep_calls and all(r == ((0, 0),) or len(r) == 1 for r in set_keep_calls)


def test_retain_at_sleep_no_shared_prefix_unchanged(tmp_path, monkeypatch):
    # Regression guard: disjoint slots/anchors still retain as before.
    hs = HoldSet(rids=("sa", "sb"), rows_by_rank=(2, 2), anchors=2, excluded=())
    monkeypatch.setattr(l15_retain, "select_hold", lambda *a, **k: hs)
    kv_buf = _kv_marker(16)
    mamba_buf = torch.arange(16 * 2, dtype=torch.float32).reshape(16, 2)

    class FakeAllocator:
        size = 16
        free_pages = torch.empty(0, dtype=torch.int64)
        release_pages = torch.empty(0, dtype=torch.int64)

        def clear(self):
            self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
            self.release_pages = torch.empty(0, dtype=torch.int64)

    candidates = [
        Candidate(rid="sa", kind="served", last_active=9.0,
                  rows_by_rank=(1, 1), anchor_depth=2, kv_depth=2),
        Candidate(rid="sb", kind="served", last_active=8.0,
                  rows_by_rank=(1, 1), anchor_depth=2, kv_depth=2),
    ]
    slots_of = {"sa": (10, 11), "sb": (12, 13)}
    anchor_of = {"sa": 14, "sb": 15}
    res = l15_retain.retain_at_sleep(
        candidates=candidates,
        node_of=lambda rid: object(),
        slots_of=lambda rid: slots_of[rid],
        anchor_slot_of=lambda rid: anchor_of[rid],
        l2_of=lambda rid: ((201, 202), (5, 6)),
        rewrite_tree=lambda node, kv_map, anchor_map, visited: None,
        caps_rows_by_rank=(4, 4),
        cap_anchor_slots=4,
        prefix=PREFIX,
        rank=1,
        epoch=77,
        pid=4242,
        kv_buffers=[kv_buf],
        mamba_buffers=[mamba_buf],
        allocator=FakeAllocator(),
        mamba_allocator=_mamba_fake(),
        reset_keep=lambda nodes: None,
        set_keep=lambda ptr, ranges: None,
        manifest_path=str(tmp_path / "l15_manifest.json"),
        log=lambda line: None,
    )
    assert res is not None
    assert len(res.manifest.spans) == 2
