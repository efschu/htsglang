"""EVICT-KEEP bind seam (review gap of 3f7378473f, 01.10.): both host pools
make their #243/#248 keep findable by the clock evict
(``HiCacheFile._arena_evict_to_disk``) when they bind an arena. Without it
the clock evict keeps only the pins and the small MAMBA arena collapses again
(z30y12 epoch 50).

Unified tree (desk/nf-release-unified-1002, 02.10.): the clock evict reads the
NF #248e seam -- ``handoff_pending.bind_arena`` in ``ArenaMHAHostPool.bind``
(role kv) and ``ArenaMambaPoolHost.bind``, read back through
``handoff_pending.pool_for_arena`` in ``HiCacheFile._arena_evict_candidates``
(stage (i) passes the pool's ``keep_for`` keys as ``keep_lo``). The 27B
backend registry (``register_keep_pool`` / ``_arena_evict_keep_lo``) is not in
the tree; the pools' duck-typed ``register_keep_pool`` call in
``ensure_bound`` stays harmless for a backend without the hook (pinned below).
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from flliper.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402


class _Backend:
    def __init__(self, register=True):
        self.arena = object()
        self.registered = []
        self.canonical_mamba_blob = types.SimpleNamespace(total_bytes=4096)
        self._canonical_kv_extents = types.SimpleNamespace(total_bytes=8192)
        self.canonical_draft_page = None
        self._kv_owner_rows = None
        if register:
            self.register_keep_pool = lambda arena, pool: self.registered.append((arena, pool))

    def _arena_for(self, total_bytes):
        return self.arena


def _pool(cls, **fields):
    pool = cls.__new__(cls)
    pool.arena = None
    pool.bound = []
    pool.bind = lambda arena, *a, **k: (pool.bound.append(arena), setattr(pool, "arena", arena))
    pool.__dict__.update(fields)
    return pool


def test_mamba_pool_registers_its_arena_at_bind():
    be = _Backend()
    pool = _pool(ArenaMambaPoolHost, _pdflip_parts=("spec", (1,), 0, 0, 1))
    assert ArenaMambaPoolHost.ensure_bound(pool, be) is True
    assert be.registered == [(be.arena, pool)]
    assert pool.bound == [be.arena] and pool._backend is be


def test_kv_pool_registers_its_arena_at_bind():
    be = _Backend()
    pool = _pool(ArenaMHAHostPool)
    assert ArenaMHAHostPool.ensure_bound(pool, be) is True
    assert be.registered == [(be.arena, pool)]


def test_a_bound_pool_does_not_register_twice():
    be = _Backend()
    pool = _pool(ArenaMambaPoolHost, _pdflip_parts=("spec", (1,), 0, 0, 1))
    ArenaMambaPoolHost.ensure_bound(pool, be)
    ArenaMambaPoolHost.ensure_bound(pool, be)
    assert len(be.registered) == 1


def test_a_backend_without_the_hook_still_binds():
    for cls, fields in ((ArenaMambaPoolHost, {"_pdflip_parts": ("spec", (1,), 0, 0, 1)}), (ArenaMHAHostPool, {})):
        be = _Backend(register=False)
        pool = _pool(cls, **fields)
        assert cls.ensure_bound(pool, be) is True
        assert pool.arena is be.arena


class _Arena:
    """Weak-referenceable stand-in recording what the clock asks for."""

    def __init__(self):
        self.asked = []

    def evict_candidates(self, want, keep_stems=(), keep_lo=None):
        self.asked.append((want, list(keep_stems), None if keep_lo is None else list(keep_lo)))
        return []


class _Pool:
    pass


def test_the_bound_pool_is_what_the_clock_evict_reads(monkeypatch):
    """bind_arena(pool, arena) -> the clock evict's stage (i) passes that
    pool's keep as keep_lo; an arena no pool is bound to keeps only the pins."""
    from flliper.srt.pdflip import handoff_pending as hp

    arena, other, pool = _Arena(), _Arena(), _Pool()
    hp.bind_arena(pool, arena)
    try:
        assert hp.pool_for_arena(arena) is pool
        assert hp.pool_for_arena(other) is None          # an unbound arena keeps nothing
        keep = type("K", (), {"keys": [7, 9], "__len__": lambda self: 2})()
        seen = []
        monkeypatch.setattr(hp, "keep_for", lambda p: (seen.append(p), keep)[1])
        HiCacheFile._arena_evict_candidates(arena, 4, 0, ["pin.sfx"])
        assert seen == [pool]
        assert arena.asked == [(4, ["pin.sfx"], [7, 9])]
        HiCacheFile._arena_evict_candidates(other, 4, 0, ["pin.sfx"])
        assert seen == [pool]                            # keep_for never asked for the unbound arena
        assert other.asked == [(4, ["pin.sfx"], None)]   # the pins-only clock
    finally:
        hp._ARENA_POOLS.pop(id(arena), None)


def test_the_real_kv_bind_makes_its_arena_findable(tmp_path):
    """The real ``ArenaMHAHostPool.bind`` (role kv, as ``ensure_bound`` calls
    it) leaves the pool findable for its arena -- the registration the 27B
    seam did in ensure_bound happens in bind on the NF form."""
    import shutil

    import pytest
    import torch

    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena
    from flliper.srt.pdflip import handoff_pending as hp

    if shutil.which("gcc") is None:
        pytest.skip("needs gcc (arena.c)")
    L, H, D, S, PAGE = 2, 2, 4, 5, 64
    win = types.SimpleNamespace(total_bytes=PAGE, extents=((8, L * 8), (PAGE // 2 + 8, L * 8)))
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = 1; p.layer_num = L; p.head_num = H; p.head_dim = D  # noqa: E702
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S  # noqa: E702
    p.element_dim = H * D; p.can_use_jit = True  # noqa: E702
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)  # noqa: E702
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, 8)
    try:
        assert hp.pool_for_arena(arena) is None
        p.bind(arena, win, role="kv", pin=False)
        assert hp.pool_for_arena(arena) is p
    finally:
        hp._ARENA_POOLS.pop(id(arena), None)
        arena.close()
