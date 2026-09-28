"""#239 S4b (F14), part 3: the one L2 (arena) under the token cut.

Form A x the token cut, host share 0 (the optimum form): the KV bytes of every
page live on the expert workers, each its own token rows of every attention
layer, the attention host (TP0) holds none. The arena host pool only knew
whole-page writers (the whole page, a PP stage's slots or a TP head slice) --
every write wanted P consecutive token ids per page, every load wrote P rows
per page into the device pool, and a worker never got the arena at all
(``get_mha_host_pool_cls`` gave it the byteless plain pool). Here a worker
writes and loads its own token rows of the shared slot, TP0 takes part in
claim/complete with no extents, and the slot completes -- becomes readable --
when the last owner has completed.

Hermetic: a tmp arena file, no CUDA. RED on e0ef2f8df5: ``bind`` takes no
owner rows, the backup refuses partial pages, ``owner_page_tokens`` does not
exist, the chooser keeps a KV-holding worker off the arena.
"""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalAbstainWindow,
    CanonicalPageWindow,
    owner_row_window,
    owner_token_runs,
)
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 2, 1, 2          # attention layers, full kv heads, head dim (uint8: cell = 2 B)
CELL = H * D
P = 4                      # tokens per page = the owner split S
BLOCK = P * CELL           # one layer's K (or V) bytes of a page
PAGE = 2 * L * BLOCK       # 32 B: [K l0 | K l1 | V l0 | V l1], token-minor
STAGING = 5
SPEC = CanonicalPageSpec(num_attn_layers=L, kv_bytes_per_token_per_attn_layer=2 * BLOCK)
WHOLE = CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=L)
W1, W2, HOST = (P, P, 0, 3), (P, P, 3, 4), (P, P, 0, 0)


def _window(owner):
    runs = owner_token_runs(*owner)
    return owner_row_window(WHOLE, P, runs) if runs else CanonicalAbstainWindow(total_bytes=PAGE)


def _pool(arena, owner):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = STAGING
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(STAGING, dtype=torch.int64)
    p.slot_used = torch.zeros(STAGING, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, STAGING, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    p.bind(arena, _window(owner), role="kv", pin=False, owner_rows=owner)
    return p


def _dev(rows, fill=None):
    """A compact device KV pool: ``rows`` token rows per layer."""
    k = [torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)]
    v = [torch.zeros(rows, H, D, dtype=torch.uint8) for _ in range(L)]
    if fill is not None:
        for l in range(L):
            for r in range(rows):
                k[l][r] = fill(0, l, r)
                v[l][r] = fill(1, l, r)
    return types.SimpleNamespace(k_buffer=k, v_buffer=v)


def _val(kv, layer, tok):
    return 10 + 50 * kv + 20 * layer + tok


@pytest.fixture
def arena(tmp_path):
    return ShmArena(str(tmp_path / "kv.bin"), PAGE, 8)


def test_owner_tokens_and_geometry(arena):
    w1, w2, h = _pool(arena, W1), _pool(arena, W2), _pool(arena, HOST)
    assert w1._owner_tok.tolist() == [0, 1, 2] and w2._owner_tok.tolist() == [3]
    assert h._owner_tok.tolist() == [] and h._own_extents == []
    # the slot geometry is the whole page on every rank
    assert w1._k_offs == [0, BLOCK] and w1._v_offs == [2 * BLOCK, 3 * BLOCK]
    assert w2._own_extents == list(_window(W2).extents)


def test_two_owners_complete_the_page_and_the_host_completes_nothing(arena):
    w1, w2, h = _pool(arena, W1), _pool(arena, W2), _pool(arena, HOST)
    slot = w1._claim(["pg0"])[0]
    assert w2._claim(["pg0"]) == [slot] and h._claim(["pg0"]) == [slot]
    base = STAGING + slot * P
    # worker 1: tokens 0..2 of the page from compact rows 0..2
    d1 = _dev(3, lambda kv, l, r: _val(kv, l, r))
    w1.backup_from_device_all_layer(d1, torch.tensor([base, base + 1, base + 2]),
                                    torch.tensor([0, 1, 2]), "direct")
    assert w1.complete_write(torch.tensor([base, base + 1, base + 2])) == 0
    # the host claims and completes with no extents: nothing written, not complete
    assert h.complete_write(torch.arange(base, base + P)) == 0
    assert arena.find_states(["pg0"]) != [2]
    # worker 2: token 3 from its compact row 0 -- the last owner completes the page
    d2 = _dev(1, lambda kv, l, r: _val(kv, l, 3))
    w2.backup_from_device_all_layer(d2, torch.tensor([base + 3]), torch.tensor([0]), "direct")
    assert w2.complete_write(torch.tensor([base + 3])) == 1
    assert arena.find_states(["pg0"]) == [2]
    page = w1._page_view[slot].view(2, L, P, H * D)
    for kv in range(2):
        for l in range(L):
            for t in range(P):
                assert page[kv, l, t].tolist() == [_val(kv, l, t)] * CELL


def test_each_owner_loads_only_its_rows_into_its_compact_pool(arena):
    w1, w2 = _pool(arena, W1), _pool(arena, W2)
    slot = w1._claim(["pg1"])[0]
    w2._claim(["pg1"])
    base = STAGING + slot * P
    w1.backup_from_device_all_layer(_dev(3, lambda kv, l, r: _val(kv, l, r)),
                                    torch.tensor([base, base + 1, base + 2]), torch.tensor([0, 1, 2]), "direct")
    w2.backup_from_device_all_layer(_dev(1, lambda kv, l, r: _val(kv, l, 3)),
                                    torch.tensor([base + 3]), torch.tensor([0]), "direct")
    back = _dev(2)
    for l in range(L):
        w2.load_to_device_per_layer(back, torch.tensor([base + 3]), torch.tensor([1]), l, "direct")
    for l in range(L):
        assert back.k_buffer[l][1].flatten().tolist() == [_val(0, l, 3)] * CELL
        assert back.v_buffer[l][1].flatten().tolist() == [_val(1, l, 3)] * CELL
        assert bool((back.k_buffer[l][0] == 0).all())  # nothing but the asked row


def test_a_row_outside_the_owner_range_is_refused(arena):
    w2 = _pool(arena, W2)
    slot = w2._claim(["pg2"])[0]
    base = STAGING + slot * P
    with pytest.raises(RuntimeError, match="ARENA-OWNER-WRITE REFUSED"):
        w2.backup_from_device_all_layer(_dev(1, lambda kv, l, r: 1), torch.tensor([base + 0]),
                                        torch.tensor([0]), "direct")


def test_ensure_bound_takes_the_owner_rows_from_the_backend(arena):
    for owner, want in ((W1, [0, 1, 2]), (HOST, [])):
        p = object.__new__(ArenaMHAHostPool)
        p.layout = "layer_first"; p.page_size = P; p.layer_num = L; p.head_num = H; p.head_dim = D
        p.dtype = torch.uint8; p.size = STAGING
        p._arena_init_fields()
        be = types.SimpleNamespace(_canonical_kv_extents=_window(owner), _kv_owner_rows=owner,
                                   _arena_for=lambda total: arena)
        p._pin = False
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("SGLANG_HICACHE_ARENA_PREPIN", "0")
            assert p.ensure_bound(be, role="kv") is True
        assert p._owner_tok.tolist() == want


def test_the_chooser_puts_a_kv_holding_worker_on_the_arena(monkeypatch):
    from sglang.srt import rank_role
    from sglang.srt.mem_cache.pool_host import mha as mha_mod

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_HOST", "1")
    dp = types.SimpleNamespace(head_dim=D, v_head_dim=D, page_size=64)
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: True)
    assert mha_mod.get_mha_host_pool_cls(dp, role="kv") is ArenaMHAHostPool
    assert mha_mod.get_mha_host_pool_cls(dp, role="draft") is mha_mod.MHATokenToKVPoolHost
    monkeypatch.setattr(rank_role, "form_a_worker_holds_kv", lambda: False)
    assert mha_mod.get_mha_host_pool_cls(dp, role="kv") is mha_mod.MHATokenToKVPoolHost


def test_a_kv_worker_refuses_a_sidecar_with_bytes():
    """A KV-holding worker has one byte-carrying host pool, the KV anchor; a
    sidecar with bytes keyed by the worker's arena KV ids is refused by name
    at attach (the #249 BYTELESS-GROW shape, where skipping is right only
    because nothing is there)."""
    from sglang.srt.managers.cache_controller import refuse_kv_worker_sidecar_bytes
    from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

    def _entry(name, spt, anchor=False):
        host = types.SimpleNamespace(size_per_token=spt, layout="layer_first", page_size=P,
                                     device="cpu", size=1)
        return PoolEntry(name=name, host_pool=host, device_pool=None, layer_mapping={},
                         is_primary_index_anchor=anchor)

    ok = HostPoolGroup([_entry("kv", 64, True), _entry("mamba", 0), _entry("qsa_indexer", 0)])
    refuse_kv_worker_sidecar_bytes(ok)
    bad = HostPoolGroup([_entry("kv", 64, True), _entry("qsa_indexer", 12)])
    with pytest.raises(RuntimeError, match="SIDECAR-BYTES REFUSED.*qsa_indexer"):
        refuse_kv_worker_sidecar_bytes(bad)
