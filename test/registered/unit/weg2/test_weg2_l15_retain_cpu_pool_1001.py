# SPDX-License-Identifier: Apache-2.0
"""L1.5 retain driven against REAL CPU pool objects (AP 1001).

test_weg2_l15_retain_0930.py pins the orchestrator against fake buffers.
This file keeps the geometry of that scenario but swaps the fakes for the
real objects the scheduler hook hands in at D's sleep:

* a CPU ``HybridReqToTokenPool`` (scaffold copied from
  test_mamba_pool_floor.py / test_mamba_reset_keep_1001.py),
* the per-layer mamba views exactly as scheduler.py builds them
  ([c[l] for c in conv ...] + [temporal[l] ...]) -- the retain must move
  bytes INSIDE the pool's own conv/temporal storage,
* the real ``mamba_allocator`` (allocator/mamba.py MambaSlotAllocator),
* a small fake KV allocator (free_pages, mirrors clear()).

Pinned properties:
(a) after retain_at_sleep, the held anchors' marker bytes sit at their new
    compact slots [1, A_H) in EVERY conv layer and in temporal;
(b) the real mamba_allocator.free_slots no longer contains the new anchor
    slots (reserve_mamba_slots carved them out);
(c) HybridReqToTokenPool.clear(keep_mamba_rows=A_H) afterwards keeps rows
    [0, A_H) (padding row 0 + the two anchors) and zeroes rows >= A_H.

Markers are (slot + 1) per slot row: nonzero everywhere, so a kept row can
never be confused with a zeroed one, and every value is exact in bfloat16.
"""

from __future__ import annotations

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.weg2 import l15_retain
from sglang.srt.weg2.l15_policy import Candidate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
MAMBA_SIZE = 6          # slots 0..6 (slot 0 = padding, rows 0..MAMBA_SIZE)
MAX_CONTEXT_LEN = 128

# KV geometry copied from test_weg2_l15_retain_0930.py: prefix (0,1,2),
# running as rank 1 -> L_H=8, rows_by_rank=(4,4), one kv move (9,5).
PREFIX = (0, 1, 2)
RANK = 1
EPOCH = 101
PID = 4343

CANDIDATES = (
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
# Held anchors well above A_H=3: 5 -> 1 and 6 -> 2 (anchor_plan squeeze).
ANCHOR_SLOT_OF = {"r_seat": 5, "r_parked": 6}
NEW_ANCHORS = (1, 2)
A_H = 3
ALLOC_SIZE = 16


def _build_pool() -> HybridReqToTokenPool:
    """CPU HybridReqToTokenPool (scaffold from test_mamba_pool_floor.py)."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)
    full_attention_layer_ids = [
        i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)
    ]
    mamba_layers = [i for i in range(NUM_LAYERS) if i not in full_attention_layer_ids]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=512,
            n_groups=4,
            num_heads=8,
            head_dim=64,
            state_size=32,
            conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    return HybridReqToTokenPool(
        size=10,
        mamba_size=MAMBA_SIZE,
        mamba_spec_state_size=10,
        max_context_len=MAX_CONTEXT_LEN,
        device="cpu",
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=False,
        enable_linear_replayssm=False,
    )


def _marker_value(slot: int) -> int:
    return slot + 1  # nonzero everywhere; exact in bfloat16


def _fill_mamba_markers(pool: HybridReqToTokenPool) -> None:
    """Every slot row of conv/temporal gets its own nonzero marker value."""
    cache = pool.mamba_pool.mamba_cache
    for conv in cache.conv:
        for s in range(int(conv.shape[1])):
            conv[:, s].fill_(_marker_value(s))
    temporal = cache.temporal
    for s in range(int(temporal.shape[1])):
        temporal[:, s].fill_(_marker_value(s))


def _mamba_views(pool: HybridReqToTokenPool):
    """The per-layer views exactly as the scheduler hook builds them."""
    cache = pool.mamba_pool.mamba_cache
    views = []
    for c in cache.conv:
        views.extend(c[i] for i in range(int(c.shape[0])))
    views.extend(cache.temporal[i] for i in range(int(cache.temporal.shape[0])))
    return views


def _kv_marker(rows: int, cols: int) -> torch.Tensor:
    return torch.tensor(
        [[i * 10 + j for j in range(cols)] for i in range(rows)],
        dtype=torch.float32,
    )


class FakeAllocator:
    """Mirrors TokenToKVPoolAllocator.clear(): free_pages = arange(1, size+1)."""

    def __init__(self):
        self.size = ALLOC_SIZE
        self.free_pages = torch.empty(0, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)


class FakeNode:
    def __init__(self):
        self.kv_slots = ()
        self.anchor_slot = -1


def _drive(tmp_path, pool):
    """Fill markers, run retain_at_sleep against the REAL pool objects."""
    _fill_mamba_markers(pool)
    kv_buffers = [_kv_marker(6, 3), _kv_marker(6, 3)]
    mamba_views = _mamba_views(pool)
    nodes = {"r_seat": FakeNode(), "r_parked": FakeNode()}
    keep_calls, reset_calls, log_lines = [], [], []

    result = l15_retain.retain_at_sleep(
        candidates=list(CANDIDATES),
        node_of=lambda rid: nodes[rid],
        slots_of=lambda rid: SLOTS_OF[rid],
        anchor_slot_of=lambda rid: ANCHOR_SLOT_OF[rid],
        l2_of=lambda rid: ((201, 202), (5, 6)),
        caps_rows_by_rank=CAPS_ROWS_BY_RANK,
        cap_anchor_slots=CAP_ANCHOR_SLOTS,
        prefix=PREFIX,
        rank=RANK,
        epoch=EPOCH,
        pid=PID,
        kv_buffers=kv_buffers,
        mamba_buffers=mamba_views,
        allocator=FakeAllocator(),
        mamba_allocator=pool.mamba_allocator,
        reset_keep=lambda ns: reset_calls.append(
            [(tuple(n.kv_slots), n.anchor_slot) for n in ns]
        ),
        set_keep=lambda buf, spans: keep_calls.append((id(buf), tuple(spans))),
        manifest_path=str(tmp_path / "l15_manifest.json"),
        log=log_lines.append,
    )
    assert result is not None, f"retain skipped: {log_lines}"
    return {
        "result": result,
        "kv_buffers": kv_buffers,
        "mamba_views": mamba_views,
        "nodes": nodes,
        "keep_calls": keep_calls,
        "reset_calls": reset_calls,
        "log_lines": log_lines,
    }


def _assert_anchor_markers(cache, rows: dict[int, int]) -> None:
    """Every conv layer and temporal slice carries the expected marker value
    at each slot row named in ``rows`` (slot -> expected marker)."""
    for conv in cache.conv:
        for l in range(int(conv.shape[0])):
            for slot, want in rows.items():
                got = conv[l, slot]
                assert torch.all(got == want), (
                    f"conv layer {l} slot {slot}: expected marker {want}, "
                    f"got {got.flatten().tolist()[:4]}..."
                )
    temporal = cache.temporal
    for l in range(int(temporal.shape[0])):
        for slot, want in rows.items():
            got = temporal[l, slot]
            assert torch.all(got == want), (
                f"temporal layer {l} slot {slot}: expected marker {want}, "
                f"got {got.flatten().tolist()[:4]}..."
            )


def test_retain_lands_held_anchor_markers_in_real_pool(tmp_path):
    """(a) The anchor moves write into the pool's OWN conv/temporal storage:
    after retain, the old slot-5 bytes sit at row 1 and the old slot-6 bytes
    at row 2, in every conv layer and in temporal; row 0 (padding) is
    untouched."""
    pool = _build_pool()
    sc = _drive(tmp_path, pool)
    assert sc["result"].a_h == A_H
    # the squeeze really moved anchors 5 and 6 down into [1, A_H)
    spans = {s.rid: s for s in sc["result"].manifest.spans}
    assert (spans["r_seat"].anchor_slot, spans["r_parked"].anchor_slot) == (1, 2)
    _assert_anchor_markers(
        pool.mamba_pool.mamba_cache,
        {0: _marker_value(0), 1: _marker_value(5), 2: _marker_value(6)},
    )
    # kv move (9,5): owner rows 4 -> 2 on this rank, in both kv buffers
    for buf in sc["kv_buffers"]:
        assert torch.equal(buf[2], _kv_marker(6, 3)[4])
    # nodes rewritten; set_keep seen on every buffer exactly once
    assert sc["nodes"]["r_seat"].anchor_slot == 1
    assert sc["nodes"]["r_parked"].anchor_slot == 2
    assert len(sc["reset_calls"]) == 1 and len(sc["reset_calls"][0]) == 2
    by_target = {}
    for buf_id, spans in sc["keep_calls"]:
        by_target.setdefault(buf_id, []).append(spans)
    for view in sc["mamba_views"]:
        assert by_target.get(id(view)) == [((0, A_H),)], "mamba view keep span"
    for buf in sc["kv_buffers"]:
        assert by_target.get(id(buf)) == [((0, 4),)], "kv buffer keep span"


def test_retain_reserves_new_anchor_slots_in_real_mamba_allocator(tmp_path):
    """(b) reserve_mamba_slots carved the new anchor slots out of the REAL
    MambaSlotAllocator: [1, A_H) are gone from free_slots, the rest keeps
    its order."""
    pool = _build_pool()
    sc = _drive(tmp_path, pool)
    free = [int(x) for x in pool.mamba_allocator.free_slots.tolist()]
    assert 1 not in free and 2 not in free, (
        f"held anchor slots still free: {free}"
    )
    assert free == [3, 4, 5, 6]


def test_clear_keep_mamba_rows_preserves_retained_anchor_rows(tmp_path):
    """(c) The sleep flush after a successful retain: clear(keep_mamba_rows=
    A_H) keeps rows [0, A_H) -- padding row 0 and the two compacted anchors
    at their moved markers -- and zeroes rows >= A_H in every conv layer and
    in temporal."""
    pool = _build_pool()
    sc = _drive(tmp_path, pool)
    pool.clear(keep_mamba_rows=sc["result"].a_h)
    cache = pool.mamba_pool.mamba_cache
    _assert_anchor_markers(
        cache,
        {0: _marker_value(0), 1: _marker_value(5), 2: _marker_value(6)},
    )
    for conv in cache.conv:
        for l in range(int(conv.shape[0])):
            assert torch.all(conv[l, A_H:] == 0), f"conv layer {l} tail not zeroed"
    for l in range(int(cache.temporal.shape[0])):
        assert torch.all(cache.temporal[l, A_H:] == 0), (
            f"temporal layer {l} tail not zeroed"
        )


def test_clear_keeps_held_anchor_slots_out_of_free_list(tmp_path):
    """The sleep flush must not un-reserve what retain reserved (L15-11c).

    clear() re-arms the mamba allocator, so WITHOUT the keep_mamba_rows
    carve-out it returns the held anchor slots [1, A_H) to the free list:
    the bytes survive reset_state(keep_rows=A_H) but the next mamba alloc
    would overwrite them. Pinned here: after the flush the free list equals
    the free list retain left behind. keep_mamba_rows=0 stays byte-
    identical (full re-arm) -- see test_mamba_reset_keep_1001.py.
    """
    pool = _build_pool()
    sc = _drive(tmp_path, pool)
    free_before = [int(x) for x in pool.mamba_allocator.free_slots.tolist()]
    assert 1 not in free_before and 2 not in free_before
    pool.clear(keep_mamba_rows=sc["result"].a_h)
    free_after = [int(x) for x in pool.mamba_allocator.free_slots.tolist()]
    assert free_after == free_before, (
        f"the flush changed the mamba free list: before={free_before} "
        f"after={free_after}"
    )
