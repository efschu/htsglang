"""W3-ARENA on a Form A D group, metal replay dual15 (45nh9g, ...09301211 @292108fda7).

What the log shows:
* The shared KV arena was full: ARENA-REF-HOLDERS pool=FULL, 720896 slots,
  every COMPLETE slot referenced.
* Each D rank logged 215 ARENA-DROP with freed=0 and l3 written:0. TP0's
  claims needed 12 pages, TP1's needed 1409, so the ranks were uneven.
* New tail leaves could not be backed (#1421 arena_claim). On a TP group they
  may not be dropped rank-locally, so they blocked device eviction
  (behind_device_child=349011) and the head request looped: SEAT-AGE DISPLACE
  trigger=kv every ~26 s.
* The W3 spill made for exactly this never ran on D: 0 "W3-ARENA SPILL"
  lines. It returned early on every R12 (Form A) rank.

The fix, R12 style:
* TP0 decides. On its own refused claim, or on a worker's refused claim left
  next to the arena, TP0 secures finished pages to L3, gives its references
  back and records SPILL.
* Every worker applies SPILL at the broadcast: it gives its own reference
  back and marks the node l3_present.
* A slot is free only once every rank's reference is gone; then the claim
  finds room.

Hermetic: the real C arena on a temp file and the real ArenaMHAHostPool, one
per rank on the SAME arena file. Real ``_weg2_direct_claim``,
``_w3_arena_spill``, ``arena_secure_to_disk`` and ``form_a_host_shadow``
attach/consume/apply. Pages of 64 B, page_size 1."""

import os
import shutil
import sys
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_w3_arena_spill_0929 as W  # noqa: E402

from sglang.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

FULL = W.FULL
SLOTS = 8


def _rank_pool(tmp_path, arena_path, store):
    p = object.__new__(ArenaMHAHostPool)
    p.layout = "layer_first"; p.page_size = 1; p.layer_num = W.L; p.head_num = W.H; p.head_dim = W.D
    p.dtype = torch.uint8; p.device = "cpu"; p.pin_memory = False; p.size = W.S
    p.element_dim = W.H * W.D; p.can_use_jit = True
    p.free_slots = torch.arange(W.S, dtype=torch.int64); p.slot_used = torch.zeros(W.S, dtype=torch.bool)
    p.kv_buffer = torch.zeros(2, W.L, W.S, W.H, W.D, dtype=torch.uint8)
    p._arena_init_fields()
    arena = ShmArena(arena_path, W.PAGE, SLOTS)
    p.bind(arena, W._Win(), role="kv", pin=False)
    p._backend = W._file_backend(store)
    return p, arena


def _tree(pool):
    t = W._tree(pool)
    t._components_tuple = (t.components[FULL],)
    t.tree_components = (FULL,)
    t._update_evictable_leaf_sets = lambda node: None
    return t


def _mirror(t, pool, arena, hashes_per_node):
    """One rank's tree over the SAME finished pages: +1 reference per page."""
    nodes, parent = [], t.root_node
    for hashes in hashes_per_node:
        stems = pool._stems(hashes)
        slots = [int(s) for s, _st in arena.find_slots(stems)]
        assert arena.ref_slots(slots, +1) == len(slots)
        rows = pool.arena_ids(torch.tensor(slots, dtype=torch.int64))
        n = W._node(t, parent, hashes, rows)
        n.key = list(hashes)
        nodes.append(n)
        parent = n
    return nodes


@pytest.fixture
def group(tmp_path):
    """TP0 and one worker (TP1) on one arena, every slot COMPLETE and held by
    BOTH trees, as on the metal (D tree=612400 of 720896, all referenced)."""
    R._reset_state_for_tests()
    store = tmp_path / "store"
    store.mkdir()
    ap = str(tmp_path / "kv.bin")
    p0, a0 = _rank_pool(tmp_path, ap, str(store))
    p1, a1 = _rank_pool(tmp_path, ap, str(store))
    hashes = [[f"a{i}"] for i in range(SLOTS)]
    pay = torch.zeros((W.PAGE,), dtype=torch.uint8)
    for i, hs in enumerate(hashes):                # every owner wrote: COMPLETE, no reference yet
        pay.fill_(0x40 + i)
        st = p0._stems(hs)
        assert a0.write(st, [W.PAGE], [((0, W.PAGE),)], [pay.data_ptr()]) == [1]
    t0, t1 = _tree(p0), _tree(p1)
    n0 = _mirror(t0, p0, a0, hashes)
    n1 = _mirror(t1, p1, a1, hashes)
    return types.SimpleNamespace(p0=p0, p1=p1, a0=a0, a1=a1, t0=t0, t1=t1, n0=n0, n1=n1, store=store)


def _claimer(t, name):
    m = UnifiedTreeNode((FULL,))
    m.parent = t.root_node
    t.root_node.children[(name,)] = m
    m.hash_value = [f"{name}0", f"{name}1"]
    m.key = list(m.hash_value)
    m.component_data[FULL].value = torch.arange(2, dtype=torch.int64)
    return m


def _as(role):
    return mock.patch.object(R, "role", lambda: role)


def test_dual15_a_workers_refused_claim_is_spilled_by_tp0_and_mirrored(group):
    g = group
    # TP1 (the rank short of room on the metal) claims and is refused -- it may
    # not spill on its own, it leaves its need for TP0
    with _as("worker"):
        m1 = _claimer(g.t1, "b")
        assert g.t1._weg2_direct_claim(m1) is False
    assert all(n.component_data[FULL].host_value is not None for n in g.n1), "a worker spilled on its own"
    # TP0's broadcast: it serves the request, spills, the SPILL events ride it
    with _as("host"):
        R.register_tree(g.t0)
        sent = R.attach([])
    assert sent and isinstance(sent[0], R.FormAHostVerdict), "TP0 sent no spill verdict"
    spills = [e for e in sent[0].events if e[0] == R.SPILL]
    assert spills, "TP0 did not spill for the worker's refused claim"
    spilled0 = [n for n in g.n0 if n.component_data[FULL].host_value is None]
    assert spilled0 and all(n.l3_present for n in spilled0)
    for n in spilled0:                               # L3 copy first, never lost
        st = n.hash_value[0] + "_sfx"
        assert (g.store / (st + ".bin")).exists()
    # the worker applies: same nodes, its reference goes back, l3_present as on TP0
    with _as("worker"):
        c = R.apply(g.t1, sent[0].events, seq=sent[0].seq)
    assert c["spill"] == len(spills) and c["missing"] == 0
    for a, b in zip(g.n0, g.n1):
        assert (a.component_data[FULL].host_value is None) == (b.component_data[FULL].host_value is None)
        assert bool(a.l3_present) == bool(b.l3_present)
        assert b.component_data[FULL].value is not None   # stays on the device
    # every rank's reference is gone -> the worker's claim finds its room now
    with _as("worker"):
        pre = g.t1._weg2_direct_claim(m1)
    assert pre is not False and pre is not None and int(pre.numel()) == 2, g.t1.refused


def test_tp0_own_refusal_spills_and_records(group):
    g = group
    with _as("host"):
        R.register_tree(g.t0)
        m0 = _claimer(g.t0, "c")
        g.t0._weg2_direct_claim(m0)                  # refused this pass: TP1 still holds the slots
        sent = R.attach([])
    assert sent and any(e[0] == R.SPILL for e in sent[0].events)
    with _as("worker"):
        R.apply(g.t1, sent[0].events, seq=sent[0].seq)
    with _as("host"):
        pre = g.t0._weg2_direct_claim(m0)
    assert pre is not False and pre is not None, g.t0.refused


def test_a_page_not_complete_is_not_spilled_on_form_a(group):
    g = group
    n = g.n0[0]
    n.hash_value = ["a0", "zz"]                     # 2 pages, only one of them complete in the arena
    with _as("host"):
        R.register_tree(g.t0)
        g.t0._w3_arena_spill(g.p0, 1)
    assert n.component_data[FULL].host_value is not None, "a node with an unfinished page left L2"


def test_off_path_unchanged_role_none(group):
    g = group
    with _as(None):
        g.t0._weg2_direct_claim(_claimer(g.t0, "d"))
    assert not R._S.ledger, "a classic boot recorded R12 events"
