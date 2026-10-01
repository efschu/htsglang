# SPDX-License-Identifier: Apache-2.0
"""L15-11a: the L1.5 retain orchestrator at D's sleep (pure + fakes).

Pins sglang.srt.weg2.l15_retain.retain_at_sleep and its two helpers:

* the STEP ORDER select -> moves -> node rewrite -> reset_keep -> clear ->
  reserve -> set_keep -> manifest (danger direction is a wrong order:
  resetting before the moves land, or reserving after set_keep, silently
  corrupts the held KV);
* buffer content follows the moves: kept rows sit at their new compact
  positions, sources otherwise untouched;
* reserved slots leave free_pages, order of the rest preserved;
* the manifest round-trips and its fingerprint equals the logged fp;
* nothing to hold -> None, and reset_keep/set_keep/manifest untouched;
* a step 1-2 planning failure returns None with a "L15-RETAIN skipped" log;
* a failure after the moves re-raises (no half state hidden).

Geometry: prefix [0, 1, 2] (two ranks, ratio 1 each, S = 2): rank 0 owns even
global slots, rank 1 owns odd. The orchestrator runs as rank 1. Rank 1's
need of 4 odd slots drives L_H = 8 while every rank-0 held slot stays below
L_H (a rank-0 slot at/above L_H would compact onto padded slot 0, which the
allocator never holds), so rank 1's slot 9 compacts to the free odd slot 5;
anchor slots {4, 0} squeeze to A_H = 2 with move 4 -> 1.
"""

from __future__ import annotations

import os
import re
import time

import pytest
import torch

from sglang.srt.weg2 import l15_retain
from sglang.srt.weg2.l15_manifest import fingerprint, read as manifest_read
from sglang.srt.weg2.l15_policy import Candidate

ALLOC_SIZE = 16
PREFIX = (0, 1, 2)
RANK = 1
EPOCH = 77
PID = 4242

CANDIDATES = (
    # deliberately unordered: selection must not depend on input order
    Candidate(rid="r_big", kind="served", last_active=9.0,
              rows_by_rank=(0, 50), anchor_depth=4, kv_depth=4),
    Candidate(rid="r_ghost", kind="served", last_active=5.0,
              rows_by_rank=(0, 1), anchor_depth=2, kv_depth=3),
    Candidate(rid="r_seat", kind="seat", last_active=2.0,
              rows_by_rank=(3, 3), anchor_depth=6, kv_depth=6),
    Candidate(rid="r_parked", kind="parked", last_active=1.0,
              rows_by_rank=(0, 1), anchor_depth=1, kv_depth=1),
)
CAPS_ROWS_BY_RANK = (4, 10)
CAP_ANCHOR_SLOTS = 5
SLOTS_OF = {"r_seat": (1, 2, 4, 6, 3, 9), "r_parked": (7,)}
# L15-11b: no anchor may sit on padding slot 0 (dummy write target for
# padded tokens); r_parked's anchor is 5, not the original 0.
ANCHOR_SLOT_OF = {"r_seat": 4, "r_parked": 5}
# expected plan under PREFIX:
#   L_H=8, rows_by_rank=(4,4), moves=((9,5),), A_H=3, anchor moves=((4,1),(5,2))
RESERVED = {1, 2, 3, 4, 5, 6, 7}
FREE_AFTER = [8, 9, 10, 11, 12, 13, 14, 15, 16]


class FakeAllocator:
    """Mirrors TokenToKVPoolAllocator.clear(): free_pages = arange(1, size+1),
    release_pages empty; the padded slot 0 is never free."""

    def __init__(self, events):
        self.size = ALLOC_SIZE
        self.events = events
        self.free_pages = torch.empty(0, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)
        self.events.append("clear")


class FakeNode:
    """Stands in for a UnifiedRadixCache node: the two attributes L15-11b
    rewrites. kv_slots is assigned once (the rewrite), so a first-set guard
    records the rewrite event without double-counting the init assignment."""

    def __init__(self, events):
        object.__setattr__(self, "_events", events)
        object.__setattr__(self, "_rewritten", False)
        object.__setattr__(self, "kv_slots", ())
        object.__setattr__(self, "anchor_slot", -1)

    def __setattr__(self, name, value):
        if name == "kv_slots" and not self._rewritten:
            self._rewritten = True
            self._events.append("nodes")
        object.__setattr__(self, name, value)


def _marker(rows, cols):
    return torch.tensor(
        [[i * 10 + j for j in range(cols)] for i in range(rows)],
        dtype=torch.float32,
    )


def make_scenario(tmp_path, events, reset_keep=None, slots_of=None):
    kv_buf = _marker(6, 3)     # 6 rows >= L_H//S*ratio = 4
    mamba_buf = _marker(8, 2)  # 8 anchor slots >= max needed
    nodes = {"r_seat": FakeNode(events), "r_parked": FakeNode(events)}
    alloc = FakeAllocator(events)
    set_keep_calls = []

    def default_reset_keep(hold_nodes):
        events.append("reset_keep")
        set_keep_calls.append(("reset_keep", len(hold_nodes)))

    def set_keep(ptr, ranges):
        events.append("set_keep")
        set_keep_calls.append(("set_keep", id(ptr), tuple(ranges)))

    path = str(tmp_path / "l15_manifest.json")
    log_lines = []
    kwargs = dict(
        candidates=list(CANDIDATES),
        node_of=lambda rid: nodes[rid],
        slots_of=slots_of or (lambda rid: SLOTS_OF[rid]),
        anchor_slot_of=lambda rid: ANCHOR_SLOT_OF[rid],
        l2_of=lambda rid: ((201, 202), (5, 6)),
        caps_rows_by_rank=CAPS_ROWS_BY_RANK,
        cap_anchor_slots=CAP_ANCHOR_SLOTS,
        prefix=PREFIX,
        rank=RANK,
        epoch=EPOCH,
        pid=PID,
        kv_buffers=[kv_buf],
        mamba_buffers=[mamba_buf],
        allocator=alloc,
        reset_keep=default_reset_keep if reset_keep is None else reset_keep,
        set_keep=set_keep,
        manifest_path=path,
        log=log_lines.append,
    )
    return {
        "kwargs": kwargs,
        "kv_buf": kv_buf,
        "mamba_buf": mamba_buf,
        "nodes": nodes,
        "alloc": alloc,
        "set_keep_calls": set_keep_calls,
        "manifest_path": path,
        "log_lines": log_lines,
        "events": events,
    }


def test_step_order_select_moves_nodes_reset_clear_reserve_setkeep_manifest(tmp_path, monkeypatch):
    events = []
    sc = make_scenario(tmp_path, events)
    real_select = l15_retain.select_hold
    real_apply = l15_retain.apply_moves
    real_reserve = l15_retain.reserve_slots
    real_write = l15_retain.manifest_write

    def w_select(*a, **k):
        events.append("select")
        return real_select(*a, **k)

    def w_apply(buffers, moves, owner_rows):
        events.append("moves")
        return real_apply(buffers, moves, owner_rows)

    def w_reserve(allocator, slots):
        events.append("reserve")
        return real_reserve(allocator, slots)

    def w_write(path, m):
        events.append("manifest")
        return real_write(path, m)

    monkeypatch.setattr(l15_retain, "select_hold", w_select)
    monkeypatch.setattr(l15_retain, "apply_moves", w_apply)
    monkeypatch.setattr(l15_retain, "reserve_slots", w_reserve)
    monkeypatch.setattr(l15_retain, "manifest_write", w_write)

    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None
    assert res.a_h == 3
    order = [e for e in events if e in (
        "select", "moves", "nodes", "reset_keep", "clear", "reserve",
        "set_keep", "manifest")]
    # apply_moves fires twice (kv + mamba); collapse consecutive dupes
    collapsed = [e for i, e in enumerate(order)
                 if i == 0 or e != order[i - 1]]
    assert collapsed == [
        "select", "moves", "nodes", "reset_keep", "clear", "reserve",
        "set_keep", "manifest",
    ], collapsed
    assert res.manifest.epoch == EPOCH and res.manifest.pid == PID
    assert res.manifest.anchor_slots == 3



def test_rows_land_at_new_slots_reserved_leave_free_pages_manifest_roundtrips(
        tmp_path):
    events = []
    sc = make_scenario(tmp_path, events)
    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None

    # every moved kv row's content sits at its new compact row (this rank)
    owner = l15_retain._owner_rows(PREFIX, RANK)
    assert res.plan.moves, "geometry must produce at least one kv move"
    for old_slot, new_slot in res.plan.moves:
        src, dst = owner(int(old_slot)), owner(int(new_slot))
        assert src is not None and dst is not None
        assert torch.equal(sc["kv_buf"][dst], _marker(src + 1, 3)[src])
    # anchor moves are not owner-sharded: mamba rows 4 -> 1 and 5 -> 2; the
    # reserved padding row 0 stays untouched (L15-11b)
    assert torch.equal(sc["mamba_buf"][1], _marker(5, 2)[4])
    assert torch.equal(sc["mamba_buf"][2], _marker(6, 2)[5])
    assert torch.equal(sc["mamba_buf"][0], _marker(1, 2)[0])
    # kept nodes were rewritten to the plan's new slots / new anchors
    for rid, node in sc["nodes"].items():
        assert node.kv_slots == tuple(res.plan.new_slots[rid])
    assert sc["nodes"]["r_seat"].anchor_slot == 1  # 4 squeezed to 1
    assert sc["nodes"]["r_parked"].anchor_slot == 2  # 5 squeezed to 2

    # reserved slots left free_pages; the rest keeps its order
    free = sc["alloc"].free_pages.tolist()
    assert set(RESERVED).isdisjoint(free)
    assert free == FREE_AFTER

    # manifest round-trips (pid_alive injected: PID 4242 is not alive here)
    m2 = manifest_read(sc["manifest_path"], pid_alive=lambda _pid: True)
    assert m2 is not None
    assert (m2.epoch, m2.pid) == (EPOCH, PID)
    assert m2.rows_by_rank == tuple(res.manifest.rows_by_rank)
    assert m2.anchor_slots == res.manifest.anchor_slots
    by_rid = {s.rid: s for s in m2.spans}
    assert set(by_rid) == {s.rid for s in res.manifest.spans}
    for s in res.manifest.spans:
        got = by_rid[s.rid]
        assert tuple(got.slots) == tuple(s.slots)
        assert got.anchor_slot == s.anchor_slot
        assert tuple(got.l2_slots) == (201, 202)
        assert tuple(got.l2_gens) == (5, 6)
    assert fingerprint(m2) == fingerprint(res.manifest)

    # the one log line carries exactly that fingerprint
    line = sc["log_lines"][-1]
    m = re.fullmatch(
        r"L15-RETAIN epoch=77 n=2 keep_rows_by_rank=(?P<rows>[\d,]+) "
        r"l_h=(?P<lh>\d+) anchors=3 fp=(?P<fp>-?\d+)", line)
    assert m is not None, line
    assert m.group("rows") == ",".join(str(x) for x in res.plan.rows_by_rank)
    assert m.group("lh") == str(res.plan.l_h)
    assert int(m.group("fp")) == fingerprint(res.manifest)


def test_nothing_to_hold_returns_none_without_touching_anything(tmp_path):
    events = []
    sc = make_scenario(tmp_path, events)
    sc["kwargs"]["candidates"] = []
    assert l15_retain.retain_at_sleep(**sc["kwargs"]) is None
    assert "reset_keep" not in events and "set_keep" not in events
    assert "clear" not in events
    assert not os.path.exists(sc["manifest_path"])
    assert sc["log_lines"] and sc["log_lines"][0].startswith("L15-RETAIN")


def test_reserve_slots_rejects_a_slot_that_is_not_free():
    alloc = FakeAllocator([])
    alloc.clear()
    with pytest.raises(ValueError, match="not free"):
        l15_retain.reserve_slots(alloc, [0])          # padded slot 0
    with pytest.raises(ValueError, match="not free"):
        l15_retain.reserve_slots(alloc, [ALLOC_SIZE + 1])  # beyond the pool


def test_set_keep_failure_reraises_and_is_not_turned_into_none(tmp_path):
    events = []
    sc = make_scenario(tmp_path, events)

    def boom(_ptr, _ranges):
        raise RuntimeError("tms keep failed")

    sc["kwargs"]["set_keep"] = boom
    with pytest.raises(RuntimeError, match="tms keep failed"):
        l15_retain.retain_at_sleep(**sc["kwargs"])
    # died at step 7: reset_keep already ran, the manifest was never written
    assert "reset_keep" in events
    assert not os.path.exists(sc["manifest_path"])
    assert not any("L15-RETAIN" in ln for ln in sc["log_lines"])


def test_reserve_slots_is_linear_for_large_batches():
    """reserve_slots must not be O(n^2). A real hold is tens of thousands of
    slots (a 64k-token request alone is 64k slots) and this runs inside D's
    sleep; the old wanted.count(s)-per-distinct-slot scan would be ~4e9 ops.
    Reserve 50_000 of 100_000 free slots well under 1 second (plain
    wall-clock, generous bound)."""
    alloc = FakeAllocator([])
    alloc.size = 100_000
    alloc.clear()
    slots = list(range(1, 50_001))
    start = time.perf_counter()
    removed = l15_retain.reserve_slots(alloc, slots)
    elapsed = time.perf_counter() - start
    assert removed == 50_000
    assert elapsed < 1.0, f"reserve_slots took {elapsed:.3f}s (expected linear)"
    # the 50k reserved slots are gone; the rest remain, in order.
    assert alloc.free_pages.tolist() == list(range(50_001, 100_001))


def test_retain_accepts_generator_candidates(tmp_path):
    """candidates may be a one-shot generator: retain_at_sleep materialises it
    once (list(candidates)) so step 8's cand_depth rebuild does not KeyError
    after the generator is exhausted -- a KeyError would land AFTER the
    buffers already moved (a half state), not as a benign skip."""
    events = []
    sc = make_scenario(tmp_path, events)
    sc["kwargs"]["candidates"] = (c for c in CANDIDATES)
    res = l15_retain.retain_at_sleep(**sc["kwargs"])
    assert res is not None
    assert {s.rid for s in res.manifest.spans} == {"r_seat", "r_parked"}
    assert os.path.exists(sc["manifest_path"])
