"""#1427s: the token-cut owners' pages must be able to COMPLETE in the arena.

z30n/z30p (NF D, Form A, token cut S=2): TP1 owns the even, TP2 the odd token
rows of every 64-token page, 12 attention layers, 512-byte cells -- one
owner's write is 2 x 12 x 32 = 768 disjoint extents of the 786432-byte page.
arena.c gave KV slots KV_IVALS=64 coverage intervals: the first owner's merge
overflowed, arena_complete answered 3 (logged as '#1427 ARENA-COMPLETE LOST
... recycled under the writer' although nothing was recycled), the page stayed
CLAIMED for ever and every page D wrote itself (decode tail, park) was
unreadable -- the wake re-computed 3.7-6.8k tokens (wake -> first token median
11.1-11.6 s).

Hermetic: a tmp arena file, no CUDA, the REAL NF-D page geometry. RED on
7704b80f98: both owners' completions come back 3, the page is never COMPLETE,
``ShmArena.ival_cap`` does not exist, overflow is not its own status.
"""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from flliper.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from flliper.srt.mem_cache.canonical_page_store import (
    CanonicalAbstainWindow,
    CanonicalPageWindow,
    owner_row_window,
    owner_token_runs,
)
from flliper.srt.mem_cache.pool_host import arena_pool as ap
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from flliper.srt.mem_cache.storage.file import hicache_arena as ha
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 12, 1, 512       # NF D: 12 attention layers, one 512-byte cell per token (uint8)
CELL = H * D
P = 64                     # tokens per page
S = 2                      # the token cut: TP1 even rows, TP2 odd rows
BLOCK = P * CELL           # one layer's K (or V) bytes of a page
PAGE = 2 * L * BLOCK       # 786432 B = arena-786432.bin
STAGING = P
SPEC = CanonicalPageSpec(num_attn_layers=L, kv_bytes_per_token_per_attn_layer=2 * BLOCK)
WHOLE = CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=L)
TP1, TP2, TP0 = (P, S, 0, 1), (P, S, 1, 2), (P, S, 0, 0)


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


def _val(kv, layer, tok):
    return (7 + 50 * kv + 20 * layer + tok) % 256


def _dev(toks):
    """A compact device KV pool: one row per owned token, in order."""
    k = [torch.zeros(len(toks), H, D, dtype=torch.uint8) for _ in range(L)]
    v = [torch.zeros(len(toks), H, D, dtype=torch.uint8) for _ in range(L)]
    for l in range(L):
        for r, t in enumerate(toks):
            k[l][r] = _val(0, l, t)
            v[l][r] = _val(1, l, t)
    return types.SimpleNamespace(k_buffer=k, v_buffer=v)


@pytest.fixture
def arena(tmp_path):
    return ShmArena(str(tmp_path / "kv.bin"), PAGE, 4)


def test_the_geometry_is_the_metal_one():
    assert PAGE == 786432
    assert len(_window(TP1).extents) == 2 * L * (P // S) == 768
    assert len(_window(TP2).extents) == 768


def test_the_kv_cap_covers_the_token_cut(arena):
    # 64 whole-page/stage intervals + one per 1 KiB of the page
    assert arena.ival_cap == 64 + PAGE // 1024 == 832


def test_the_token_cut_owners_complete_the_page_and_it_is_readable(arena):
    """The metal path: TP0 (share 0) claims first with no extents, TP1 and
    TP2 join, each writes its own interleaved rows, the second completion
    makes the page COMPLETE and every byte is where the loader reads it."""
    h, w1, w2 = _pool(arena, TP0), _pool(arena, TP1), _pool(arena, TP2)
    slot = h._claim(["pg"])[0]
    assert w1._claim(["pg"]) == [slot] and w2._claim(["pg"]) == [slot]
    base = STAGING + slot * P
    even, odd = list(range(0, P, 2)), list(range(1, P, 2))
    w1.backup_from_device_all_layer(_dev(even), torch.tensor([base + t for t in even]),
                                    torch.arange(len(even)), "direct")
    assert w1.complete_write(torch.tensor([base + t for t in even])) == 0   # merged, not full
    assert arena.find_states(["pg"]) != [2]
    assert h.complete_write(torch.arange(base, base + P)) == 0
    w2.backup_from_device_all_layer(_dev(odd), torch.tensor([base + t for t in odd]),
                                    torch.arange(len(odd)), "direct")
    assert w2.complete_write(torch.tensor([base + t for t in odd])) == 1   # the last owner completes
    assert arena.find_states(["pg"]) == [2]
    page = w1._page_view[slot].view(2, L, P, CELL)
    for kv in range(2):
        for l in (0, 5, L - 1):
            for t in (0, 1, 31, 62, 63):
                assert int(page[kv, l, t, 0]) == _val(kv, l, t)
                assert int(page[kv, l, t, CELL - 1]) == _val(kv, l, t)
    assert ap._COMPLETE_LOST[5] == 0


def test_overflow_is_its_own_status_and_is_named(tmp_path):
    a = ShmArena(str(tmp_path / "small.bin"), 4096, 2)
    cap = a.ival_cap
    assert cap == 64 + 4
    (s, st, g), = a.claim_slots(["x"], [4096])
    assert st == 0
    # cap + 1 disjoint one-byte extents: the coverage list cannot hold them
    ext = [(2 * i, 1) for i in range(cap + 1)]
    assert a.complete_slots([s], [g], ext) == [5]
    before = dict(ap._COMPLETE_LOST)
    assert ap._note_complete_lost([5, 1, 3], [s, 1, 2], "t") == 2
    assert ap._COMPLETE_LOST[5] == before[5] + 1 and ap._COMPLETE_LOST[3] == before[3] + 1


def test_a_recycled_slot_still_answers_3(tmp_path):
    a = ShmArena(str(tmp_path / "r.bin"), 4096, 2)
    (s, st, g), = a.claim_slots(["q"], [4096])
    a.free_slots([s], reason="test_recycle")
    assert a.complete_slots([s], [g], [(0, 4096)]) == [3]


def test_bind_refuses_an_arena_whose_cap_cannot_hold_the_cut(tmp_path):
    class Narrow(ShmArena):
        ival_cap = 64   # the pre-#1427s KV cap

    small = Narrow(str(tmp_path / "n.bin"), PAGE, 2)
    with pytest.raises(RuntimeError, match="#1427s OWNER-ROWS COVERAGE REFUSED.*need=768 cap=64"):
        _pool(small, TP1)


def test_every_free_names_its_reason(arena):
    (s, _, g), = arena.claim_slots(["f"], [PAGE])
    arena.free_slots([s], reason="unit_reason")
    assert ha._FREE_BY_REASON["unit_reason"][0] >= 1
    (s2, st2, g2), = arena.claim_slots(["f2"], [PAGE])
    assert st2 == 0
    n0 = ha._FREE_BY_REASON.get("unit_release", [0, 0])[1]
    assert arena.release_claims([s2], [g2], reason="unit_release") == [0]
    assert ha._FREE_BY_REASON["unit_release"][1] == n0 + 1
