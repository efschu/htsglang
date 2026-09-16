"""#1427 Stufe 4: the host copy is written straight from the card into the
arena slot. Hermetic: tmp arena, no CUDA, the all-layer kernel is recorded."""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.pool_host import arena_pool as ap
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 2, 2, 4
CELL = H * D
PAGE = 64
S = 5


class _Win:
    total_bytes = PAGE
    extents = ((1 * CELL, L * CELL), (PAGE // 2 + 1 * CELL, L * CELL))


class _WinOther:  # the other layer shard of the same page: cell 0 of 4 (a covers 1..2)
    total_bytes = PAGE
    extents = ((0, 1 * CELL), (PAGE // 2, 1 * CELL))


class _Backend:
    def __init__(self):
        self.evicted = 0
    def _get_suffixed_key(self, h):
        return f"{h}.sfx"
    def _arena_evict_to_disk(self, arena, want):
        self.evicted += 1
        return 0


def _pool(arena, win=_Win, role="kv", layers=L):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = 1; p.layer_num = layers; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = True; p.token_stride_size = CELL
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    import threading
    p.lock = threading.Lock()
    p.kv_buffer = torch.zeros(2, layers, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    p.bind(arena, win(), role=role, pin=False)
    p._backend = _Backend()
    return p


def _record(monkeypatch):
    calls = []
    monkeypatch.setattr(ap, "jit_transfer_hicache_all_layer", lambda **kw: calls.append(kw))
    return calls


def _device_pool():
    return types.SimpleNamespace(
        k_buffer=[torch.zeros(1, dtype=torch.uint8)], k_data_ptrs=torch.tensor([1, 2]),
        v_data_ptrs=torch.tensor([3, 4]))


def test_claim_dma_and_complete_across_two_layer_shards(tmp_path, monkeypatch):
    calls = _record(monkeypatch)
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, 8)
    a, b = _pool(arena), _pool(arena, _WinOther, layers=1)
    a._own_extents = [(1 * CELL, 3 * CELL), (PAGE // 2 + 1 * CELL, 3 * CELL)]  # a: cells 1..3
    rows = a.alloc_write(["h0", "h1"])
    assert rows is not None and all(a.is_arena_id(int(r)) for r in rows)
    slots = (rows - S).tolist()
    assert [st for _, st in arena.find_slots(["h0.sfx", "h1.sfx"])] == [1, 1]   # CLAIMED
    # the second shard joins the same slots
    rows_b = b.alloc_write(["h0", "h1"])
    assert rows_b.tolist() == rows.tolist()
    # the backup hands the kernel the SLOTS as host indices, the arena ptrs, the page stride
    a.backup_from_device_all_layer(_device_pool(), rows, torch.tensor([10, 11]), "kernel")
    kw = calls[-1]
    assert kw["indices_dst"].tolist() == slots and kw["indices_src"].tolist() == [10, 11]
    assert kw["kv_cache_dst_stride_bytes"] == PAGE and kw["kv_cache_src_stride_bytes"] == CELL
    assert kw["k_ptr_dst"].tolist() == [a._data_base + 1 * CELL + l * CELL for l in range(L)]
    # ack of shard a: merged, not full; ack of shard b: COMPLETE for everyone
    assert a.complete_write(rows) == 0
    assert [st for _, st in arena.find_slots(["h0.sfx"])] == [1]
    assert b.complete_write(rows_b) == 2
    assert [st for _, st in arena.find_slots(["h0.sfx", "h1.sfx"])] == [2, 2]
    assert not a._pending and not b._pending
    # both nodes hold a reader reference; free releases it
    assert arena.ref_slots(slots, -1) == 2 and arena.ref_slots(slots, -1) == 2
    assert a.free(rows) == 2


def test_a_second_claim_of_a_complete_page_writes_nothing(tmp_path, monkeypatch):
    calls = _record(monkeypatch)
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, 8)
    a = _pool(arena, layers=L)
    a._own_extents = [(0, PAGE // 2), (PAGE // 2, PAGE // 2)]   # this pool covers the whole page
    rows = a.alloc_write(["h0"]); a.backup_from_device_all_layer(_device_pool(), rows, torch.tensor([1]), "kernel")
    assert a.complete_write(rows) == 1 and len(calls) == 1
    rows2 = a.alloc_write(["h0"])
    assert rows2.tolist() == rows.tolist() and not a._pending
    a.backup_from_device_all_layer(_device_pool(), rows2, torch.tensor([1]), "kernel")
    assert len(calls) == 1, "a COMPLETE page is not rewritten"
    assert a.complete_write(rows2) == 0


def test_abort_frees_fresh_claims_and_a_full_arena_is_refused(tmp_path, monkeypatch):
    _record(monkeypatch)
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, 4)
    a = _pool(arena)
    rows = a.alloc_write(["x0", "x1"])
    a.abort_write(rows)
    assert not a._pending and arena.find_slots(["x0.sfx"]) == [(-1, 0)]
    assert a.alloc_write(["y0", "y1", "y2", "y3"]) is not None
    assert a.alloc_write(["z0"]) is None and a._backend.evicted == 1


def test_draft_role_maps_kv_rows_to_draft_slots(tmp_path, monkeypatch):
    calls = _record(monkeypatch)
    kv = ShmArena(str(tmp_path / "kv.bin"), PAGE, 8)
    dr = ShmArena(str(tmp_path / "draft.bin"), PAGE, 8)
    a = _pool(kv); d = _pool(dr, role="draft")
    rows = a.alloc_write(["h0", "h1"])
    assert d.alloc_write_draft(rows, ["h0", "h1"], "eagle")
    dslots = [s for s, _ in dr.find_slots(["h0.eagle.sfx", "h1.eagle.sfx"])]
    assert all(s >= 0 for s in dslots) and [d.row_slot[int(r) - S] for r in rows] == dslots
    d.backup_from_device_all_layer(_device_pool(), rows, torch.tensor([7, 8]), "kernel")
    assert calls[-1]["indices_dst"].tolist() == dslots
    d._own_extents = [(0, PAGE // 2), (PAGE // 2, PAGE // 2)]
    assert d.complete_write(rows) == 2
    assert [st for _, st in dr.find_slots(["h0.eagle.sfx"])] == [2]


def test_an_unbound_pool_skips_foreign_arena_ids_instead_of_indexing_past_staging(tmp_path, monkeypatch):
    """#1427g (xsn192): the draft pool, unbound on P, received the KV arena ids."""
    calls = []
    monkeypatch.setattr(ap.MHATokenToKVPoolHost, "backup_from_device_all_layer",
                        lambda self, dp, hi, di, io: calls.append((hi.tolist(), di.tolist())))
    p = object.__new__(ArenaMHAHostPool)
    p.size = S; p.page_size = 1
    p._arena_init_fields()
    p.backup_from_device_all_layer(None, torch.tensor([1, S + 7, S + 8]), torch.tensor([10, 11, 12]), "kernel")
    assert calls == [([1], [10])]


def test_available_size_reports_the_arena_not_the_staging_fallback(tmp_path, monkeypatch):
    """#1440 (xsn200): the front's D-seat gate serialised the parked prompts
    because available_size() answered with the 1.5k staging rows."""
    _record(monkeypatch)
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, 8)
    p = _pool(arena)
    assert p.staging_free() == S
    assert p.available_size() == 8
    rows = p.alloc_write(["a", "b"])
    assert p.available_size() == 6
    # staging alloc stays bounded by the staging free list
    assert p.alloc(S + 1) is None
    assert p.alloc(1) is not None and p.staging_free() == S - 1
