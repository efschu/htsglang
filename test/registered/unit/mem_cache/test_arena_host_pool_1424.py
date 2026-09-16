"""#1424 Stufe 3: the host tier IS the arena. Hermetic: a tmp arena file, no
CUDA; the transfer kernel is replaced by a recorder."""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.pool_host.arena_pool import PLACEHOLDERS, ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 2, 2, 4          # local layers, heads, head_dim (uint8 cells: cell = 8 B)
CELL = H * D
PAGE = 64                  # canonical page: K [4 slots x 8 B] | V [4 slots x 8 B] -> 64 B
S = 5                      # staging rows


class _Win:
    total_bytes = PAGE
    # this rank owns canonical slots 1..2 of 4: K at 8.., V at 32+8..
    extents = ((1 * CELL, L * CELL), (PAGE // 2 + 1 * CELL, L * CELL))


def _pool(tmp_path, role="kv", slots=8):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = 1; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / f"{role}.bin"), PAGE, slots)
    p.bind(arena, _Win(), role=role, pin=False)
    return p, arena


def _write_page(arena, stem, fill):
    pay = torch.full((PAGE,), fill, dtype=torch.uint8)
    assert arena.write([stem], [PAGE], [((0, PAGE),)], [pay.data_ptr()]) == [1]


def _recorder(p):
    calls = []
    def _t(device_pool, k_src, v_src, src_idx, dst_idx, layer_id):
        calls.append((k_src, src_idx.tolist(), dst_idx.tolist(), layer_id))
    p._transfer = _t
    return calls


def test_kv_pool_id_space_and_in_place_read(tmp_path, monkeypatch):
    from sglang.srt.mem_cache.pool_host import mha as mha_mod
    staged = []
    monkeypatch.setattr(mha_mod.MHATokenToKVPoolHost, "load_to_device_per_layer",
                        lambda self, dp, hi, di, layer, io: staged.append((hi.tolist(), di.tolist(), layer)))
    p, arena = _pool(tmp_path)
    assert p.id_space == S + 8 + PLACEHOLDERS and p.prefetch_capacity_tokens == 8
    ph = p.alloc_read(3)
    assert all(p.is_placeholder(int(i)) for i in ph)
    _write_page(arena, "pg0", 7); _write_page(arena, "pg1", 9)
    slots = [s for s, _ in arena.find_slots(["pg0", "pg1"])]
    host = ph.clone()
    p.resolve_rows(host, slots)
    assert [p.is_arena_id(int(i)) for i in host] == [True, True, False]
    # the strided ref reads THIS rank's K bytes of the page: slot 1 = offset 8
    k0 = p.arena_k_refs[0][slots[0]]
    assert k0.flatten().tolist() == [7] * CELL
    assert p.arena_v_refs[1][slots[1]].flatten().tolist() == [9] * CELL
    # a load splits staging rows from arena rows and hands the kernel SLOTS
    calls = _recorder(p)
    p.load_to_device_per_layer(None, torch.tensor([host[0].item(), 2, host[1].item()]), torch.tensor([10, 11, 12]), 1, "kernel")
    arena_calls = [c for c in calls if c[0] is p.arena_k_refs[1]]
    assert arena_calls and arena_calls[0][1] == slots and arena_calls[0][2] == [10, 12]
    assert staged == [([2], [11], 1)], "the staging row takes the ordinary path"
    # writes and page reads never target arena ids
    with pytest.raises(RuntimeError):
        p.get_data_page(int(host[0]))
    with pytest.raises(RuntimeError):
        p.backup_from_device_all_layer(None, host[:2], torch.tensor([0, 1]), "kernel")
    # free releases the reader references (slot evictable again), ignores placeholders
    arena.ref_slots(slots, +1)
    assert p.free(host) == 2
    assert p.free(ph) == 0


def test_draft_pool_maps_rows_to_draft_slots_and_zero_rows(tmp_path):
    p, arena = _pool(tmp_path, role="draft")
    _write_page(arena, "d0", 3)
    slot = arena.find_slots(["d0"])[0][0]
    p.resolve_draft_rows([4, 5], [slot, -1])
    calls = _recorder(p)
    rows = torch.tensor([S + 4, S + 5])
    p.load_to_device_per_layer(None, rows, torch.tensor([20, 21]), 0, "kernel")
    kinds = {("zero" if c[0] is p._zero_k else "arena"): c for c in calls}
    assert kinds["zero"][2] == [21] and kinds["arena"][1] == [slot] and kinds["arena"][2] == [20]
    assert p.free(rows) == 2 and p.row_slot == {}


class _Backend:
    def __init__(self, arena):
        self._canonical_kv_extents = _Win()
        self.canonical_draft_page = _Win()
        self._a = arena

    def _arena_for(self, total_bytes):
        return self._a

    def _get_suffixed_key(self, key):
        return key + "_sfx"


class _Op:
    def __init__(self):
        self.completed_tokens = 0

    def increment(self, n):
        self.completed_tokens += n
        return True


def test_controller_resolves_pages_in_place_without_copy(tmp_path):
    p, arena = _pool(tmp_path)
    p.arena = None  # bind lazily through the backend
    be = _Backend(arena)
    _write_page(arena, "h0_sfx", 1); _write_page(arena, "h1_sfx", 2)
    c = object.__new__(HiCacheController)
    c.mem_pool_host = p; c.storage_backend = be; c.page_size = 1
    op = _Op()
    host = p.alloc_read(4)
    n = c._arena_page_get(op, ["h0", "h1", "h2", "h3"], host)
    assert n == 2 and op.completed_tokens == 2
    slots = [s for s, _ in arena.find_slots(["h0_sfx", "h1_sfx"])]
    assert [int(host[0]) - S, int(host[1]) - S] == slots
    assert p.is_placeholder(int(host[2])), "the miss and its tail stay placeholders"
    assert arena.evict_candidates(8) == [], "both pages carry a reader reference"
    # the draft branch: rows behind the KV ids map to DRAFT slots or misses
    dp, darena = _pool(tmp_path, role="draft")
    dp.arena = None
    _write_page(darena, "h0.draft-x_sfx", 5)
    c.mem_pool_host_draft = dp
    c._draft_component_name = lambda: "draft-x"
    c._draft_l3_hits = 0; c._draft_l3_misses = 0
    dbe = _Backend(darena); dbe.canonical_draft_page = _Win()
    c.storage_backend = dbe
    flags = c._draft_page_get_generic(["h0", "h1"], host[:2])
    assert flags == [True, False] and c._draft_l3_hits == 1 and c._draft_l3_misses == 1
    assert dp.row_slot == {slots[0]: darena.find_slots(["h0.draft-x_sfx"])[0][0], slots[1]: -1}


def test_backup_ack_rebinds_staging_rows_to_arena_slots(tmp_path):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    p, arena = _pool(tmp_path)
    be = _Backend(arena)
    _write_page(arena, "n0_sfx", 1); _write_page(arena, "n1_sfx", 2)
    released = []
    cc = types.SimpleNamespace(mem_pool_host=p, storage_backend=be, mem_pool_host_draft=None,
                               append_host_mem_release=lambda host_indices, **k: released.append(host_indices.tolist()))
    t = object.__new__(UnifiedRadixCache)
    t.cache_controller = cc; t.page_size = 1
    from sglang.srt.mem_cache import unified_radix_cache as urc
    cd = types.SimpleNamespace(host_value=torch.tensor([3, 4]))
    node = types.SimpleNamespace(id=7, hash_value=["n0", "n1"], component_data={urc.BASE_COMPONENT_TYPE: cd})
    assert t._weg2_rebind_host_to_arena(node) is True
    slots = [s for s, _ in arena.find_slots(["n0_sfx", "n1_sfx"])]
    assert cd.host_value.tolist() == [S + s for s in slots]
    assert released == [[3, 4]], "the staging rows are freed"
    assert arena.evict_candidates(8) == [], "reader references hold the pages"
    assert t._weg2_rebind_host_to_arena(node) is True, "idempotent on arena rows"
    # a page missing from the arena keeps the old rows (and takes no reference)
    cd2 = types.SimpleNamespace(host_value=torch.tensor([1]))
    node2 = types.SimpleNamespace(id=8, hash_value=["gone"], component_data={urc.BASE_COMPONENT_TYPE: cd2})
    assert t._weg2_rebind_host_to_arena(node2) is False and cd2.host_value.tolist() == [1]
