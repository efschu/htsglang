"""W3-ARENA (kvs2 boot 09291223, bs2 x 240k, P PP0 12:30:55): the shared KV
arena (5461 slots) was full and EVERY COMPLETE slot carried the P tree's reader
reference (ARENA-REF-HOLDERS tree=5419 tree_in_use=0). The claim's room-making
(#1427 ``_evict_for_claim``) takes unreferenced slots only, and the #257 L3
write covers only what it takes -- so 74751 claims in a row freed nothing and
wrote nothing to L3 (``ARENA-DROP freed=0 ... l3=on_disk:0,written:0``). The
next node's backup was refused (#1421 arena_claim, then parent_unbacked down
the chain), /flush_cache answered 400 and the flip ended in W3.

The fix: under claim pressure the tree spills finished pages -- each gets its
L3 copy first, then the node's Full host rows go (the node stays in the tree
and on the device, ``l3_present``) -- and the claim finds its room.

Hermetic: the real C arena on a temp file, the real ``alloc_write`` /
``complete_write`` / ``_evict_for_claim`` / ``arena_secure_to_disk`` and the
real ``UnifiedRadixCache._weg2_direct_claim``; pages of 64 B, page_size 1."""

import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    UnifiedRadixCache,
    UnifiedTreeNode,
)

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

L, H, D = 2, 2, 4
CELL = H * D
PAGE = 64
S = 5
FULL = ComponentType.FULL


class _Win:
    total_bytes = PAGE
    extents = ((1 * CELL, L * CELL), (PAGE // 2 + 1 * CELL, L * CELL))


class _Evictor:
    def __init__(self, refuse=()):
        self.refuse = set(refuse)

    def reserve(self, stem, size, key=None, owner_writes_whole_file=False):
        return stem not in self.refuse

    def commit(self, stem):
        pass

    def abort(self, stem):
        pass


def _file_backend(root, refuse=()):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._sharded_path = _path
    be._ensure_shard_dir = lambda path: None
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._evictor = _Evictor(refuse)
    be._key_geom = {"is_mla_model": False}
    be._arena_evict_to_disk = lambda arena, want: 0
    be._get_suffixed_key = lambda key: key + "_sfx"
    be._suffix_for_key = lambda key: ("_sfx",)
    return be


def _pool(tmp_path, slots, refuse=()):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = 1; p.layer_num = L; p.head_num = H; p.head_dim = D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = S
    p.element_dim = H * D; p.can_use_jit = True
    p.free_slots = torch.arange(S, dtype=torch.int64); p.slot_used = torch.zeros(S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, L, S, H, D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(str(tmp_path / "kv.bin"), PAGE, slots)
    p.bind(arena, _Win(), role="kv", pin=False)
    root = tmp_path / "store"
    root.mkdir(exist_ok=True)
    p._backend = _file_backend(str(root), refuse)
    return p, arena, root


def _finished(p, arena, hashes, fill):
    """A finished page as the metal leaves it: COMPLETE (every PP rank's
    extents in) and one reader reference per page held by the tree -- the
    ack's ``complete_write`` +1."""
    stems = p._stems(hashes)
    pay = torch.full((PAGE,), fill & 0xFF, dtype=torch.uint8)
    for st in stems:
        assert arena.write([st], [PAGE], [((0, PAGE),)], [pay.data_ptr()]) == [1]
    slots = [int(s) for s, _st in arena.find_slots(stems)]
    assert arena.ref_slots(slots, +1) == len(slots)
    return p.arena_ids(torch.tensor(slots, dtype=torch.int64))


class _NoLRU:
    def in_list(self, node):
        return False


class _FullComp:
    """The Full component's host half, as full_component.evict_component
    frees it: the pool's own free (arena references go back), host_value None."""

    component_type = FULL

    def __init__(self, pool):
        self.pool = pool

    def evict_component(self, node, target=EvictLayer.DEVICE):
        cd = node.component_data[FULL]
        host_freed = 0
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            self.pool.free(cd.host_value)
            cd.host_value = None
        return 0, host_freed


def _tree(pool):
    t = object.__new__(UnifiedRadixCache)
    t.root_node = UnifiedTreeNode((FULL,))
    t.page_size = 1
    t.ongoing_write_through = {}
    t.evictable_host_leaves = set()
    t._r12_rec = None
    t.lru_lists = {FULL: _NoLRU()}
    t.host_lru_lists = {FULL: _NoLRU()}
    t.components = {FULL: _FullComp(pool)}
    t.cache_controller = types.SimpleNamespace(mem_pool_host=pool, mem_pool_host_draft=None)
    t.refused = []
    t._1421_refused = lambda why, node: t.refused.append(why)
    t._weg2_direct_pool = lambda: pool
    return t


def _node(t, parent, hashes, rows):
    n = UnifiedTreeNode((FULL,))
    n.parent = parent
    parent.children[tuple(hashes)] = n
    n.hash_value = list(hashes)
    n.component_data[FULL].value = torch.arange(len(hashes), dtype=torch.int64)
    n.component_data[FULL].host_value = rows
    n.l3_present = True   # the direct-write ack marks it store-present (#1427)
    return n


def _full_tree(tmp_path, slots=8, refuse=()):
    """The metal's shape: every slot COMPLETE and referenced by a node whose
    KV is still on the device (bs2 x 240k: request 1 finished, request 2
    prefilling)."""
    p, arena, root = _pool(tmp_path, slots, refuse)
    t = _tree(p)
    nodes = []
    parent = t.root_node
    for i in range(slots):
        rows = _finished(p, arena, [f"a{i}"], 0x40 + i)
        parent = _node(t, parent, [f"a{i}"], rows)
        nodes.append(parent)
    return p, arena, root, t, nodes


def _claimer(t):
    m = UnifiedTreeNode((FULL,))
    m.parent = t.root_node
    t.root_node.children[("b",)] = m
    m.hash_value = ["b0", "b1"]
    m.component_data[FULL].value = torch.arange(2, dtype=torch.int64)
    return m


def test_kvs2_shape_a_full_arena_of_tree_held_pages_frees_nothing_on_its_own(tmp_path):
    """The metal line, reproduced: every COMPLETE slot carries the tree's
    reference -> the claim's room-making frees 0 and writes 0 to L3."""
    p, arena, root, t, nodes = _full_tree(tmp_path)
    freed = p._evict_for_claim(arena, 2, claim_stem="b0_sfx")
    assert freed == 0
    assert not list(root.iterdir())


def test_kvs2_shape_the_claim_spills_finished_pages_to_l3_and_gets_its_room(tmp_path):
    """RED on 424346f693: _weg2_direct_claim refuses (#1421 arena_claim) and
    not one page reaches L3 -- the W3 chain of the kvs2 boot. GREEN: the
    finished pages go to L3 first, the tree hands its references back, the
    claim gets its two slots; the spilled nodes stay in the tree and on the
    device (l3_present), and every spilled page reads back from disk."""
    p, arena, root, t, nodes = _full_tree(tmp_path)
    dropped_before = getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0)
    m = _claimer(t)
    pre = t._weg2_direct_claim(m)
    assert pre is not False, f"claim refused: {t.refused}"
    assert pre is not None and int(pre.numel()) == 2
    spilled = [n for n in nodes if n.component_data[FULL].host_value is None]
    assert spilled, "no node gave its host copy back"
    for n in spilled:
        assert n.component_data[FULL].value is not None   # still on the device
        assert n.l3_present
        st = n.hash_value[0] + "_sfx"
        assert (root / (st + ".bin")).read_bytes() == bytes([0x40 + int(n.hash_value[0][1:])]) * PAGE
    # the claim's room took two of them out of L2: those live on disk only,
    # every other one is still COMPLETE in the arena -- none is lost
    stems = [n.hash_value[0] + "_sfx" for n in spilled]
    gone = [st for st, (slot, _s) in zip(stems, arena.find_slots(stems)) if slot < 0]
    assert len(gone) == 2
    assert all((root / (st + ".bin")).exists() for st in gone)
    assert getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0) == dropped_before


def test_a_page_without_an_l3_copy_is_never_released(tmp_path):
    """A node whose page could not be written to L3 keeps its host copy
    (never leaves L2 without an L3 copy, #257 (d)); the others spill."""
    p, arena, root, t, nodes = _full_tree(tmp_path, refuse=("a0_sfx",))
    m = _claimer(t)
    t._weg2_direct_claim(m)
    assert nodes[0].component_data[FULL].host_value is not None
    assert not (root / "a0_sfx.bin").exists()
    assert any(n.component_data[FULL].host_value is None for n in nodes[1:])


def test_a_pending_write_and_a_host_locked_load_are_not_spilled(tmp_path):
    p, arena, root, t, nodes = _full_tree(tmp_path)
    t.ongoing_write_through[nodes[1].id] = object()
    nodes[2].component_data[FULL].host_lock_ref = 1
    m = _claimer(t)
    t._weg2_direct_claim(m)
    assert nodes[1].component_data[FULL].host_value is not None
    assert nodes[2].component_data[FULL].host_value is not None


def test_the_claimers_own_chain_and_host_only_nodes_stay(tmp_path):
    """The claimer's ancestors are the chain its parent rule reads; a node
    whose KV left the device has its host copy as its only copy on P."""
    p, arena, root, t, nodes = _full_tree(tmp_path)
    nodes[3].component_data[FULL].value = None     # evicted from the device
    chain_parent = nodes[5]
    m = UnifiedTreeNode((FULL,))
    m.parent = chain_parent
    chain_parent.children[("b",)] = m
    m.hash_value = ["b0", "b1"]
    m.component_data[FULL].value = torch.arange(2, dtype=torch.int64)
    t._weg2_direct_claim(m)
    for anc in nodes[:6]:
        if anc is nodes[3]:
            continue
        assert anc.component_data[FULL].host_value is not None, anc.hash_value
    assert nodes[3].component_data[FULL].host_value is not None
