"""EVICT-KEEP bind seam (review gap of 3f7378473f, 01.10.): both host pools
register themselves with the storage backend when they bind an arena, so the
clock evict (``HiCacheFile._arena_evict_to_disk``) can find their #243/#248
keep. Without the registration ``_arena_evict_keep_lo`` answers None and the
small MAMBA arena collapses again (z30y12 epoch 50).

The registration is duck-typed (``getattr(..., "register_keep_pool")``): a
backend without it still binds.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402


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
    pool = _pool(ArenaMambaPoolHost, _weg2_parts=("spec", (1,), 0, 0, 1))
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
    pool = _pool(ArenaMambaPoolHost, _weg2_parts=("spec", (1,), 0, 0, 1))
    ArenaMambaPoolHost.ensure_bound(pool, be)
    ArenaMambaPoolHost.ensure_bound(pool, be)
    assert len(be.registered) == 1


def test_a_backend_without_the_hook_still_binds():
    for cls, fields in ((ArenaMambaPoolHost, {"_weg2_parts": ("spec", (1,), 0, 0, 1)}), (ArenaMHAHostPool, {})):
        be = _Backend(register=False)
        pool = _pool(cls, **fields)
        assert cls.ensure_bound(pool, be) is True
        assert pool.arena is be.arena


def test_the_registered_pool_is_what_the_clock_evict_reads(monkeypatch):
    from sglang.srt.weg2 import handoff_pending as hp

    store = HiCacheFile.__new__(HiCacheFile)
    arena, other, pool = object(), object(), object()
    store.register_keep_pool(arena, pool)
    keep = type("K", (), {"keys": [7, 9], "__len__": lambda self: 2})()
    seen = []
    monkeypatch.setattr(hp, "keep_for", lambda p: (seen.append(p), keep)[1])
    assert store._arena_evict_keep_lo(arena) == [7, 9]
    assert seen == [pool]
    assert store._arena_evict_keep_lo(other) is None   # an unregistered arena keeps nothing
