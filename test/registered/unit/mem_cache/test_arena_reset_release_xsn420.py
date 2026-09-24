"""weg2xsn420: a tree reset gives the tree's arena references back, and a
claim that needs room frees it without disk I/O.

THE DEFECT (27B line 3dbb79013f; identical on eb5d04453f, xsn408-411): an
arena host pool keeps one reader reference per row a tree node holds
(`complete_write` +1) and drops it in `free` (-1). The flush reset replaced the
tree and called `mem_pool_host.clear()`, which resets only the staging
bookkeeping -- so every published anchor stayed referenced, no slot of a phase
ever became an eviction candidate, and the 112-slot mamba arena was full after
one 4x100k needle phase. Every later publish was refused (`mamba_claim`), D's
store read answered zero, and the request ended W50 -> requeue -> W53 -> 413.

USER RULE 24.09.: HiCache work and Mamba anchors never impair a running
prefill or decode -- no wait, sync or copy in the compute path. The release is
host bookkeeping at the reset's idle points (one numpy pass per pool, no
wait); a claim that finds no free slot now frees exactly what it needs in C,
with NO disk write (the old evict round wrote up to 256 pages -- for the mamba
arena 256 x 74.8 MiB -- in the scheduler thread; it only never fired because
nothing was evictable).

Hermetic, CPU. REAL C arena (built into $TMPDIR), REAL ArenaMambaPoolHost /
ArenaMHAHostPool methods, REAL UnifiedRadixCache release:
  * shipped reset (clear only): the next claim is refused -- the boot's
    statuses=[4] -- and nothing is an eviction candidate;
  * releasing reset: the next claim succeeds, freeing exactly one slot, and
    the backend's evict-to-disk is never called;
  * rows an in-flight host operation uses are left alone, pending rows are
    not released, one failing pool never breaks the reset;
  * a non-arena host pool is not touched (default path unchanged).
"""

import threading
import types

import pytest
import torch

from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentData,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

SLOTS, PAGE, STAGING = 8, 4096, 3


def _arena_or_skip(path):
    from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

    try:
        return ShmArena(str(path), PAGE, SLOTS)
    except RuntimeError as exc:  # the C helper could not be built here
        pytest.skip(f"arena helper unavailable: {exc}")


class _NoDiskBackend:
    """The claim must never reach the disk round any more."""

    def __init__(self):
        self.disk_rounds = 0

    def _arena_evict_to_disk(self, arena, want):
        self.disk_rounds += 1
        raise AssertionError("claim-time eviction wrote to disk in the compute path")


def _pool(path):
    from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost

    pool = object.__new__(ArenaMambaPoolHost)
    pool.size = STAGING
    pool._arena_init_fields()
    pool.device = "cpu"
    pool.lock = threading.RLock()
    pool.page_size = 1
    pool.mem_state = torch.zeros((STAGING,), dtype=torch.uint8)
    pool.free_slots = torch.arange(STAGING, dtype=torch.int64)
    pool.arena = _arena_or_skip(path)
    pool.arena_slots = SLOTS
    pool.id_space = STAGING + SLOTS
    pool._backend = _NoDiskBackend()
    pool._own_extents = [(0, PAGE)]
    pool._page_bytes = PAGE
    return pool


def _publish(pool, stem):
    slots = pool._claim([stem])
    if slots is None:
        return None
    host = torch.tensor([STAGING + s for s in slots], dtype=torch.int64)
    pool.complete_write(host)
    return host


def _node(children=(), host=None, pending=None, host_lock=0):
    data = [ComponentData() for _ in ComponentType]
    data[ComponentType.MAMBA].host_value = host
    data[ComponentType.MAMBA].host_lock_ref = host_lock
    return types.SimpleNamespace(children={i: c for i, c in enumerate(children)},
                                 component_data=data, write_through_pending_id=pending)


def _cache(pool, nodes, arena=True):
    comp = types.SimpleNamespace(component_type=ComponentType.MAMBA, _mamba_pool_host=pool)
    return types.SimpleNamespace(
        cache_controller=types.SimpleNamespace(mem_pool_host=types.SimpleNamespace(arena_read=arena)),
        root_node=_node(children=nodes), _components_tuple=(comp,))


def _fill_phase(pool):
    hosts = [_publish(pool, f"needle-anchor-{i}") for i in range(SLOTS)]
    assert all(h is not None for h in hosts)
    return hosts


def _clear(pool):
    type(pool).__mro__[1].clear(pool)      # the flush: MambaPoolHost.clear


def test_the_shipped_reset_leaks_every_reference_and_the_arena_stays_full(tmp_path):
    pool = _pool(tmp_path / "a.bin")
    _fill_phase(pool)
    _clear(pool)
    assert pool.arena.evict_candidates(SLOTS) == []
    assert _publish(pool, "next-request-anchor") is None   # xsn420: statuses=[4], mamba_claim


def test_the_releasing_reset_lets_the_next_phase_publish_without_disk_io(tmp_path):
    pool = _pool(tmp_path / "b.bin")
    hosts = _fill_phase(pool)
    cache = _cache(pool, [_node(host=h) for h in hosts])
    assert UnifiedRadixCache._release_host_values_before_reset(cache) == SLOTS
    _clear(pool)
    before = pool.arena.stats()["complete"]
    assert _publish(pool, "next-request-anchor") is not None
    assert pool.arena.stats()["complete"] == before     # exactly one old anchor made room
    assert pool._backend.disk_rounds == 0


def test_rows_an_inflight_operation_uses_are_left_alone(tmp_path):
    pool = _pool(tmp_path / "c.bin")
    hosts = _fill_phase(pool)
    nodes = [_node(host=hosts[0], pending=17),     # write-through in flight
             _node(host=hosts[1], host_lock=1)]    # a host op holds the rows
    nodes += [_node(host=h) for h in hosts[2:]]
    assert UnifiedRadixCache._release_host_values_before_reset(_cache(pool, nodes)) == SLOTS - 2
    # the two in-use anchors keep their reference: never a candidate
    kept = {int(hosts[0][0]) - STAGING, int(hosts[1][0]) - STAGING}
    assert kept.isdisjoint({c[0] for c in pool.arena.evict_candidates(SLOTS)})


def test_pending_rows_are_not_released(tmp_path):
    pool = _pool(tmp_path / "d.bin")
    slots = pool._claim(["claimed-not-acked"])           # pending: the copy is in flight
    assert pool.release_tree_rows(torch.tensor([STAGING + slots[0]])) == 0
    assert slots[0] in pool._pending


def test_one_failing_pool_never_breaks_the_reset():
    class _Bad:
        def release_tree_rows(self, rows):
            raise AssertionError("inconsistent pool")

    cache = _cache(_Bad(), [_node(host=torch.tensor([5]))])
    assert UnifiedRadixCache._release_host_values_before_reset(cache) == 0


def test_a_non_arena_host_pool_and_a_first_reset_are_untouched():
    calls = []

    class _Pool:
        def release_tree_rows(self, rows):
            calls.append(rows)
            return 1

    cache = _cache(_Pool(), [_node(host=torch.tensor([1]))], arena=False)
    assert UnifiedRadixCache._release_host_values_before_reset(cache) == 0 and calls == []
    fresh = types.SimpleNamespace(_components_tuple=())            # __init__: no tree yet
    assert UnifiedRadixCache._release_host_values_before_reset(fresh) == 0


def test_the_reset_releases_before_it_replaces_the_tree_and_the_claims_do_no_disk_io():
    import inspect

    from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost
    from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

    src = inspect.getsource(UnifiedRadixCache._reset_full)
    assert src.index("self._release_host_values_before_reset()") < src.index(
        "self.root_node = UnifiedTreeNode(")
    for fn in (ArenaMHAHostPool._claim, ArenaMHAHostPool._claim_np):
        body = inspect.getsource(fn)
        assert "_arena_evict_to_disk" not in body and "_evict_for_claim" in body
    assert ArenaMambaPoolHost._evict_for_claim is ArenaMHAHostPool._evict_for_claim
    assert ArenaMambaPoolHost.release_tree_rows is ArenaMHAHostPool.release_tree_rows
