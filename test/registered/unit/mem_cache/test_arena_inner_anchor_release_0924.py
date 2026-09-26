"""Group P gives INNER mamba anchors back once the chain moved past them, and
holds the END anchors one phase across the flip's reset (27B line, on A's
reset release 479f6eccb0).

MEASURED NEED (weg2xsn420): P writes one anchor per written node -- per
4096-token chunk (mamba_checkpoint_interval None = per node, the user's law
since 27./28.08) plus the #1481 N-1 split -- and held a reference on every one
until the reset: one 4x98k phase took ~102 of the 112 mamba slots, 6x262k
would take 384. The hand-back needs the #1481 end anchor per prompt; every
other anchor is prefix cache and may be dropped by a claim (A's
`_evict_for_claim`, no disk I/O).

Hermetic, CPU, the REAL C arena and REAL pool / tree / component methods:
  * the chain: completing node k releases the anchor of node k-1 (tombstoned
    in the tree first, then the reference), never an `_weg2_end_anchor`, a
    host-locked or a write-pending one; the released anchor stays COMPLETE and
    findable, and a full arena's claim drops it -- never the held end anchor;
  * group D and the switch off release nothing;
  * the reset holds the end anchors one phase: an empty reset (the second
    flush of the same sleep, an idle flip) keeps the hold, the next non-empty
    reset gives it back.
"""

import threading
import types

import pytest
import torch

from sglang.srt.mem_cache.unified_cache_components import (
    ComponentData,
    ComponentType,
    EvictLayer,
    MambaComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    INNER_ANCHOR_RELEASE_ENV,
    UnifiedRadixCache,
)

SLOTS, PAGE, STAGING = 8, 4096, 3


@pytest.fixture(autouse=True)
def _group_p(monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv(INNER_ANCHOR_RELEASE_ENV, "1")    # the arm arms it (default off)
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", True)   # the launcher sets it on P


def test_the_default_is_off(monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.delenv(INNER_ANCHOR_RELEASE_ENV)
    assert urc._weg2_inner_anchor_release_on() is False


def test_without_the_end_anchor_mark_nothing_is_released(monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    assert urc._weg2_inner_anchor_release_on() is True
    monkeypatch.setattr(urc, "_WEG2_END_ANCHOR", False)
    assert urc._weg2_inner_anchor_release_on() is False


def _arena_or_skip(path):
    from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

    try:
        return ShmArena(str(path), PAGE, SLOTS)
    except RuntimeError as exc:  # the C helper could not be built here
        pytest.skip(f"arena helper unavailable: {exc}")


class _NoDiskBackend:
    def _arena_evict_to_disk(self, arena, want):
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


class _Lru:
    def __init__(self):
        self.nodes = set()

    def in_list(self, node):
        return id(node) in self.nodes

    def remove_node(self, node):
        self.nodes.discard(id(node))

    def insert_mru(self, node):
        self.nodes.add(id(node))


class _Mamba:
    """The REAL MambaComponent.evict_component over this pool and a host LRU."""

    component_type = ComponentType.MAMBA
    evict_component = MambaComponent.evict_component

    def __init__(self, pool):
        self._mamba_pool_host = pool
        self.cache = types.SimpleNamespace(host_lru_lists={ComponentType.MAMBA: _Lru()})


def _node(parent=None, host=None, *, end=False, lock=0, pending=None):
    data = [ComponentData() for _ in ComponentType]
    data[ComponentType.MAMBA].host_value = host
    data[ComponentType.MAMBA].host_lock_ref = lock
    n = types.SimpleNamespace(id=id(data), parent=parent, children={}, component_data=data,
                              write_through_pending_id=pending)
    if end:
        n._weg2_end_anchor = True
    if parent is not None:
        parent.children[len(parent.children)] = n
    return n


def _tree(pool):
    root = _node()
    comp = _Mamba(pool)
    t = types.SimpleNamespace(
        root_node=root, components={ComponentType.MAMBA: comp}, _components_tuple=(comp,),
        cache_controller=types.SimpleNamespace(mem_pool_host=types.SimpleNamespace(arena_read=True)))
    t._weg2_carrier_hold = types.MethodType(UnifiedRadixCache._weg2_carrier_hold, t)
    return t, root


def _release(t, node, pool):
    return UnifiedRadixCache._weg2_release_inner_anchor(t, node, pool)


def _slot(host):
    return int(host[0]) - STAGING


def test_the_chain_releases_the_anchor_it_moved_past(tmp_path):
    pool = _pool(tmp_path / "a.bin")
    t, root = _tree(pool)
    h = [_publish(pool, f"chunk-{i}") for i in range(3)]
    n1 = _node(root, h[0]); n2 = _node(n1, h[1]); n3 = _node(n2, h[2])
    assert _release(t, n2, pool) is True               # chunk 2 acked: chunk 1 is inner
    assert n1.component_data[ComponentType.MAMBA].host_value is None, "tombstoned in the tree"
    assert _release(t, n3, pool) is True               # chunk 3 acked: chunk 2 is inner
    assert n2.component_data[ComponentType.MAMBA].host_value is None
    assert n3.component_data[ComponentType.MAMBA].host_value is not None, "the chain's end is kept"
    # released anchors stay COMPLETE and findable by stem -- prefix cache
    assert [st for _, st in pool.arena.find_slots(["chunk-0", "chunk-1", "chunk-2"])] == [2, 2, 2]
    cands = {c[0] for c in pool.arena.evict_candidates(SLOTS)}
    assert cands == {_slot(h[0]), _slot(h[1])}, "exactly the two inner anchors are droppable"


def test_the_hand_back_anchor_and_in_use_anchors_are_kept(tmp_path):
    pool = _pool(tmp_path / "b.bin")
    t, root = _tree(pool)
    h = [_publish(pool, f"a-{i}") for i in range(4)]
    end = _node(root, h[0], end=True)
    assert _release(t, _node(end, h[1]), pool) is False
    locked = _node(root, h[2], lock=1)
    assert _release(t, _node(locked, h[3]), pool) is False
    assert end.component_data[ComponentType.MAMBA].host_value is not None
    assert locked.component_data[ComponentType.MAMBA].host_value is not None
    assert pool.arena.evict_candidates(SLOTS) == [], "every anchor still referenced"


def test_a_fork_keeps_its_anchor(tmp_path):
    """User decision 24.09. point 2, "Forks bleiben" (added with the per-path
    cap, Agent G): an anchor whose node has two children serves both branches
    -- the chain moving past it on ONE branch does not make it inner."""
    pool = _pool(tmp_path / "g.bin")
    t, root = _tree(pool)
    h = [_publish(pool, f"f-{i}") for i in range(3)]
    fork = _node(root, h[0])
    _node(fork, h[1])                                   # branch 1
    fork._weg2_fork = True                              # set by the insert of branch 2
    assert _release(t, _node(fork, h[2]), pool) is False  # branch 2's anchor acked
    assert fork.component_data[ComponentType.MAMBA].host_value is not None
    assert pool.arena.evict_candidates(SLOTS) == [], "the fork's anchor stays referenced"


def test_a_child_only_one_rank_has_does_not_make_a_fork(tmp_path):
    """On group P only PP0 reads the store: a host-prefetched second child may
    hang below a node on PP0 alone. The fork is what the DEVICE insert marked
    (same step on every rank), not `len(children)` -- else PP0 would keep an
    anchor its peers release."""
    pool = _pool(tmp_path / "h.bin")
    t, root = _tree(pool)
    h = [_publish(pool, f"g-{i}") for i in range(2)]
    n1 = _node(root, h[0])
    _node(n1, None)                                     # PP0's prefetched host chain
    assert _release(t, _node(n1, h[1]), pool) is True
    assert n1.component_data[ComponentType.MAMBA].host_value is None


def test_a_full_arena_drops_released_inner_anchors_never_the_end_anchor(tmp_path):
    pool = _pool(tmp_path / "c.bin")
    t, root = _tree(pool)
    hs = [_publish(pool, f"p-{i}") for i in range(SLOTS)]   # the arena is full
    end = _node(root, hs[0], end=True)
    prev = end
    for h in hs[1:]:
        prev = _node(prev, h)
        _release(t, prev, pool)
    # the end anchor and the chain's last anchor stay referenced; 6 inner ones are droppable
    for i in range(SLOTS - 2):
        assert _publish(pool, f"next-{i}") is not None, "a claim makes room from inner anchors"
    assert _publish(pool, "one-too-many") is None, "the referenced end anchors are never dropped"
    assert [st for _, st in pool.arena.find_slots(["p-0"])] == [2]


@pytest.mark.parametrize("group,switch", [("D", None), ("P", "0")])
def test_group_d_and_the_switch_off_release_nothing(tmp_path, monkeypatch, group, switch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", group)
    if switch is not None:
        monkeypatch.setenv(INNER_ANCHOR_RELEASE_ENV, switch)
    pool = _pool(tmp_path / "d.bin")
    t, root = _tree(pool)
    h = [_publish(pool, f"d-{i}") for i in range(2)]
    n1 = _node(root, h[0])
    assert _release(t, _node(n1, h[1]), pool) is False
    assert n1.component_data[ComponentType.MAMBA].host_value is not None


def test_the_reset_holds_the_end_anchors_one_phase(tmp_path):
    pool = _pool(tmp_path / "e.bin")
    t, root = _tree(pool)
    he, hx = _publish(pool, "end"), _publish(pool, "inner")
    _node(root, he, end=True); _node(root, hx)
    UnifiedRadixCache._release_host_values_before_reset(t)
    assert {c[0] for c in pool.arena.evict_candidates(SLOTS)} == {_slot(hx)}, \
        "phase k: the end anchor is held across the reset"
    # the second flush of the same sleep (and an idle flip) resets an EMPTY tree
    t.root_node = _node()
    UnifiedRadixCache._release_host_values_before_reset(t)
    assert pool.arena.evict_candidates(SLOTS) == [], "an empty reset keeps the hold"
    # phase k+1 (D drained in between, #1011): the next non-empty reset gives it back
    hy = _publish(pool, "phase-k+1")
    t.root_node = _node()
    _node(t.root_node, hy)
    UnifiedRadixCache._release_host_values_before_reset(t)
    assert {c[0] for c in pool.arena.evict_candidates(SLOTS)} == {_slot(he), _slot(hy)}


def test_group_d_resets_as_a_releases_them(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    pool = _pool(tmp_path / "f.bin")
    t, root = _tree(pool)
    he = _publish(pool, "end")
    _node(root, he, end=True)
    UnifiedRadixCache._release_host_values_before_reset(t)
    assert {c[0] for c in pool.arena.evict_candidates(SLOTS)} == {_slot(he)}


def test_the_ack_path_calls_the_release_after_the_complete():
    import inspect

    src = inspect.getsource(UnifiedRadixCache._weg2_direct_complete)
    assert src.index("mp.complete_write(mhv)") < src.index("self._weg2_release_inner_anchor(node, mp)")
    assert EvictLayer.HOST  # the release goes through the component's own host eviction
