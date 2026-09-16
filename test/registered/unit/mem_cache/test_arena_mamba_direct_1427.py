"""#1427 Stufe 4b: the mamba anchor host pool lives in the mamba arena --
strided views on this rank's layer/segment extents, a backup is a device->
view index copy, a load the reverse, prefetch resolves COMPLETE blobs in
place. Hermetic: tmp arena, CPU tensors as the 'device' pool."""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.hicache_migrate import MambaBlobSpec, conv_extents, temporal_extents
from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

SPEC = MambaBlobSpec(num_layers=2, num_heads=2, head_dim=2, state_size=2, conv_dim=6, conv_width=2,
                     key_dim=2, value_dim=2, units=2, temporal_itemsize=4, conv_itemsize=2)
TOTAL = SPEC.temporal_bytes + SPEC.conv_bytes   # 64 + 48
S = 3        # staging (old anchor) slots
N = 4        # device slots


class _Backend:
    def __init__(self, arena):
        self._arena = arena
        self.canonical_mamba_blob = types.SimpleNamespace(total_bytes=TOTAL)
        self.evicted = 0
    def _arena_for(self, total):
        return self._arena
    def _get_suffixed_key(self, k):
        return f"{k}.sfx"
    def _log_key(self, pool, k):
        return f"{k}.{pool.value if hasattr(pool, 'value') else pool}"
    def _arena_evict_to_disk(self, arena, want):
        self.evicted += 1
        return 0


def _device_pool(fill=True):
    L = SPEC.num_layers
    t = torch.arange(L * N * 8, dtype=torch.float32).reshape(L, N, 2, 2, 2) if fill else torch.zeros(L, N, 2, 2, 2)
    c = torch.arange(L * N * 12, dtype=torch.float16).reshape(L, N, 6, 2) if fill else torch.zeros(L, N, 6, 2, dtype=torch.float16)
    return types.SimpleNamespace(mamba_cache=types.SimpleNamespace(temporal=t, conv=[c]), num_mamba_layers=L, size=N)


def _pool(arena, ratios=(1,), rank=0, t_shape=(2, 2, 2), conv_shape=(6, 2)):
    p = object.__new__(ArenaMambaPoolHost)
    p.size = S; p.page_size = 1; p.layout = "layer_first"; p.device = "cpu"; p.pin_memory = False
    p.num_mamba_layers = SPEC.num_layers
    p.temporal_state_shape = t_shape; p.temporal_state_elem_size = int(torch.tensor(t_shape).prod())
    p.conv_state_shapes = [conv_shape]; p.conv_state_elem_sizes = [int(torch.tensor(conv_shape).prod())]
    p.temporal_dtype = torch.float32; p.conv_dtype = torch.float16; p.dtype = torch.float16
    p.free_slots = torch.arange(S, dtype=torch.int64)
    p.temporal_buffer = torch.zeros(SPEC.num_layers, S, 2, 2, 2)
    p.conv_buffer = [torch.zeros(SPEC.num_layers, S, 6, 2, dtype=torch.float16)]
    p._arena_init_fields()
    p._weg2_parts = (SPEC, list(ratios), rank, 0, SPEC.num_layers)
    be = _Backend(arena)
    assert p.ensure_bound(be)
    return p


def _slot_bytes(arena, slot):
    return bytes(arena.slot_view(slot, TOTAL))


def test_backup_lands_in_the_blob_at_the_store_cut_and_loads_back(tmp_path):
    arena = ShmArena(str(tmp_path / "mamba.bin"), TOTAL, 8)
    p = _pool(arena)
    dev = _device_pool()
    rows = p.alloc_write(["h9"])
    assert rows is not None and p.is_arena_id(int(rows[0]))
    slot = int(rows[0]) - S
    p.backup_from_device_all_layer(dev, rows, torch.tensor([3]), "direct")
    blob = _slot_bytes(arena, slot)
    t_ext = temporal_extents(SPEC, [1], 0)
    c_ext = conv_extents(SPEC, [1], 0)
    for l in range(SPEC.num_layers):
        off, ln = t_ext[l]
        assert blob[off:off + ln] == dev.mamba_cache.temporal[l][3].contiguous().view(torch.uint8).numpy().tobytes()
        row = dev.mamba_cache.conv[0][l][3]
        ch0 = 0
        for j in range(3):
            off, ln = c_ext[3 * l + j]
            n = ln // (2 * 2)
            assert blob[off:off + ln] == row[ch0:ch0 + n].contiguous().view(torch.uint8).numpy().tobytes()
            ch0 += n
    # the ack completes the page (this pool covers every extent), reference taken
    assert p.complete_write(rows) == 1
    assert arena.find_slots(["h9.mamba.sfx"]) == [(slot, 2)]
    # a load from the arena rebuilds the device rows exactly
    back = _device_pool(fill=False)
    for l in range(SPEC.num_layers):
        p.load_to_device_per_layer(back, rows, torch.tensor([1]), l, "direct")
        assert torch.equal(back.mamba_cache.temporal[l][1], dev.mamba_cache.temporal[l][3])
        assert torch.equal(back.mamba_cache.conv[0][l][1], dev.mamba_cache.conv[0][l][3])
    # the staging rows still take the old path (untouched here)
    assert p.free(rows) == 1 and arena.ref_slots([slot], -1) == 0


def test_prefetch_resolves_a_complete_blob_in_place_and_skips_the_copy(tmp_path):
    arena = ShmArena(str(tmp_path / "mamba.bin"), TOTAL, 8)
    p = _pool(arena)
    rows = p.alloc_write(["h1"]); p.backup_from_device_all_layer(_device_pool(), rows, torch.tensor([0]), "direct")
    p.complete_write(rows)
    ph = p.alloc_read(2)
    assert all(p.is_placeholder(int(i)) for i in ph)
    flags = p.arena_resolve_reads(p._backend, ph, ["h1.mamba", "h2.mamba"])
    assert flags == [True, None]
    assert int(ph[0]) == int(rows[0]) and p.is_placeholder(int(ph[1]))
    with pytest.raises(RuntimeError):
        p.set_from_flat_data_page(int(ph[0]), torch.zeros(TOTAL, dtype=torch.uint8))


def test_two_head_shards_join_one_blob(tmp_path):
    arena = ShmArena(str(tmp_path / "mamba.bin"), TOTAL, 8)
    a = _pool(arena, ratios=(1, 1), rank=0, t_shape=(1, 2, 2), conv_shape=(3, 2))
    b = _pool(arena, ratios=(1, 1), rank=1, t_shape=(1, 2, 2), conv_shape=(3, 2))
    ra = a.alloc_write(["k"]); rb = b.alloc_write(["k"])
    assert ra.tolist() == rb.tolist()
    half = types.SimpleNamespace(mamba_cache=types.SimpleNamespace(
        temporal=torch.ones(2, N, 1, 2, 2), conv=[torch.ones(2, N, 3, 2, dtype=torch.float16)]))
    a.backup_from_device_all_layer(half, ra, torch.tensor([0]), "direct")
    assert a.complete_write(ra) == 0                     # half the heads in: not complete
    b.backup_from_device_all_layer(half, rb, torch.tensor([0]), "direct")
    assert b.complete_write(rb) == 1                     # both shards in: COMPLETE
    assert arena.find_slots(["k.mamba.sfx"])[0][1] == 2
