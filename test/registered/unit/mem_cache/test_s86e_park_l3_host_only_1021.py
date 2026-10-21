"""S86e PARK-L3 (rc27 boot ..._893d09fa3f_1021_023225, D group TP=3 Form A): the sleep release keeps no host-only
node and no mamba-aux node whose copies L3 holds -- the park-end chain gets in before the reset.

THE METAL (D log). 02:44:15 ``S86 SLEEP-L3-RELEASE n=3 ... vetoed_by_group=6251 group_released=16..183
kept=host_only:19,aux_host_only:7`` beside ``S83 VICTIM-CENSUS ... pinned=6418 tree_device=2237 tree_host_only=4014``;
then ``#1470 FLUSH-UNBACKED site=flush nodes=4 tokens=10112`` and ``S85 PARK-END-UNBACKED rid=pdflip-2-30 node=155
depth=139008 store_end=133632 lost_tokens=5376``; P recomputed them (``PDFLIP-CLAIM-SHORT rid=pdflip-2-30 ...
by=kv_prefix tokens=133632``). 03:05:42 the same (``kept=host_only:14,aux_host_only:8``, ``tree_host_only=4480``,
``#1470 FLUSH-PUBLISH ARENA-FULL-STOP unbacked=14 arena_refused=14 issued=0``).

Hermetic like the S86 suite: the real C arena on a temp file, one real ``ArenaMHAHostPool`` per rank on the SAME file,
the real ``secure_rows_to_l3`` (the write-behind's form via the conftest seam), the real ``_pdflip_direct_claim``,
``form_a_host_shadow`` roles, two ranks in threads with an element-wise MAX behind a barrier. The tree's host-leaf
removal is the tree's own predicate (``_is_host_leaf``) with the removal reduced to what this harness's list keys allow
(the Full component's ALL eviction gives the arena references back, the node leaves its parent)."""
import os
import sys
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_s86_sleep_l3_release_1020 as B  # noqa: E402
import test_w3_arena_spill_0929 as W  # noqa: E402
import test_w3_arena_spill_forma_dual15_0930 as D2  # noqa: E402
from test_s86_sleep_l3_release_1020 import _thread_roles  # noqa: E402,F401

from flliper.srt.managers import pdflip_flush_nonblock as F  # noqa: E402
from flliper.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import EvictLayer  # noqa: E402
from flliper.srt.pdflip import sleep_l3_release as S86  # noqa: E402

FULL, SLOTS = W.FULL, 8
DEV = 4            # a0..a3 stay on the device (n[0..3]); a4..a7 are host-only (n[4..7], a chain, n[7] the leaf)


def _host_leaf_removal(t):
    """``_evict_host_leaf`` for the list-keyed harness tree: the tree's own predicate, the Full component's ALL
    eviction (the arena references go back), the node leaves its parent."""
    def ehl(node, tracker):
        assert t._is_host_leaf(node), "S86e removed a node that is no host leaf"
        _d, hf = t._evict_component_and_detach_lru(node, t.components[FULL], target=EvictLayer.ALL, tracker=None)
        tracker[FULL] = tracker.get(FULL, 0) + int(hf or 0)
        par = node.parent
        key = next(k for k, v in par.children.items() if v is node)
        par.children.pop(key)
        t.evictable_host_leaves.discard(node)
    t._evict_host_leaf = ehl
    t.tree_components = (FULL,)


@pytest.fixture
def rc27(tmp_path):
    """TP0 and a worker on ONE arena (8 slots, every one COMPLETE and held by BOTH trees, every page in L3). The
    first four nodes are on the device and a load reads them on the worker (host lock -- the arena's device half is
    vetoed, as rc27's aux_host_only nodes were); the last four are HOST-ONLY (the metal's tree_host_only). Under the
    last device node the park-end chain of pdflip-2-30: two nodes with no copy anywhere."""
    R._reset_state_for_tests()
    g = B._group(tmp_path)
    for n in g.n0:
        sec = g.p0.secure_rows_to_l3(n.component_data[FULL].host_value)
        assert sec["lost"] == 0 and sec["pages"] == 1
    for t, nodes in ((g.t0, g.n0), (g.t1, g.n1)):
        _host_leaf_removal(t)
        for n in nodes[DEV:]:
            n.component_data[FULL].value = None              # demoted: host-only (evicted, backuped)
    for n in g.n1[:DEV]:
        n.component_data[FULL].host_lock_ref = 1             # the device half is kept by the group
    g.c0 = B._chain(g.t0, g.n0[DEV - 1], ["c0", "c1"], "pdflip-2-30")
    g.c1 = B._chain(g.t1, g.n1[DEV - 1], ["c0", "c1"], "pdflip-2-30")
    return g


def _alive(t):
    return [n for n in t._collect_all_nodes() if n is not t.root_node]


# ---- (1) the metal, composed --------------------------------------------------------------------------------------

def test_rc27_host_only_tree_releases_its_l3_pages_and_the_park_end_chain_gets_in(rc27):
    """RED on a773b4ebc4 (S86 vetoes every host-only node: ``kept=host_only:4,host_locked:4``, the claims refused,
    S85 PARK-END-UNBACKED lost_tokens=2). GREEN: the host-only subtree leaves on both ranks, lost 0."""
    g = rc27
    assert B._refs(g) == [2] * SLOTS
    (r0, r1), _grp = B._sleep_release(g)
    assert r0["agreed"] and r1["agreed"]
    assert r0["pages"] == r1["pages"] == SLOTS - DEV, (r0, r1)
    assert B._refs(g) == [2] * DEV + [0] * (SLOTS - DEV)
    for t, nodes in ((g.t0, g.n0), (g.t1, g.n1)):
        alive = _alive(t)
        assert all(n not in alive for n in nodes[DEV:]), "a host-only node of an L3-held subtree stayed"
        assert all(n in alive for n in nodes[:DEV])
    B._flush_round(g)
    lost = B._lost(g.t0)
    assert lost == 0, "S85 PARK-END-UNBACKED lost_tokens=%d" % lost


def test_the_old_veto_is_named_and_the_switch_off_keeps_everything(rc27):
    """Switch off (the S86 step not run): the chain is lost -- TP0's own W3 spill (K23) lets go of TP0's references
    of the host-only nodes, the worker's stay (its request waits for a broadcast the flush never sends)."""
    g = rc27
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_SLEEP_L3_RELEASE": "0"}):
        B._sleep_release(g)
        B._flush_round(g)
    assert B._lost(g.t0) == 2
    assert all(r >= 1 for r in B._refs(g)), B._refs(g)
    assert all(n in _alive(g.t1) for n in g.n1), "a worker let go without the step"


# ---- (2) never a page without its copy (danger: the sleep frees an unsecured page silently) ------------------------

def test_a_host_only_page_without_an_l3_copy_keeps_its_node_and_every_ancestor(rc27):
    """Mutant guard: a5 has no L3 copy. Its node stays on BOTH ranks, and so do its host-only ancestor a4 (it cannot
    leave before its child) -- named ``host_only:not_on_l3`` / ``host_only:child_kept``. The leaves below that hold
    copies (a6, a7) leave; nothing names a5 on disk."""
    g = rc27
    st = g.n0[5].hash_value[0] + "_sfx"
    os.remove(g.store / (st + ".bin"))
    g.p0._backend._evictor.refuse.add(st)
    g.p1._backend._evictor.refuse.add(st)
    (r0, r1), _grp = B._sleep_release(g)
    assert r0["agreed"] and r1["agreed"]
    refs = B._refs(g)
    assert refs[5] == 2 and refs[4] == 2, refs
    assert refs[6] == refs[7] == 0, refs
    for t, nodes in ((g.t0, g.n0), (g.t1, g.n1)):
        alive = _alive(t)
        assert nodes[4] in alive and nodes[5] in alive
        assert nodes[5].component_data[FULL].host_value is not None
    assert r0["why"].get("host_only:not_on_l3") == 1 and r0["why"].get("host_only:child_kept") == 1, r0["why"]
    assert not (g.store / (st + ".bin")).exists()


def test_a_kept_descendant_on_one_rank_keeps_the_whole_subtree_on_every_rank(rc27):
    """Rank unity through the subtree rule: the worker's a7 is read by a load (host lock). TP0 has no reason of its
    own -- the fixed point keeps a7 AND its host-only ancestors a4..a6 on TP0 too, the digest agrees, residual 0."""
    g = rc27
    g.n1[7].component_data[FULL].host_lock_ref = 1
    (r0, r1), grp = B._sleep_release(g)
    assert r0["agreed"] and r1["agreed"] and r0["residual"] == r1["residual"] == 0
    assert B._refs(g) == [2] * SLOTS
    for t, nodes in ((g.t0, g.n0), (g.t1, g.n1)):
        assert all(n in _alive(t) for n in nodes)
    assert grp.calls[0] == grp.calls[1]


def test_a_kept_leaf_on_one_rank_lets_the_other_leaves_go(rc27):
    """A sibling leaf with its copy still leaves: a4 gets a second host-only child b8 holding nothing kept; only the
    kept branch (a5..a7 on the worker: a5 host-locked) and a4 stay."""
    g = rc27
    g.n1[5].component_data[FULL].host_lock_ref = 1
    (r0, r1), _grp = B._sleep_release(g)
    assert r0["agreed"] and r1["agreed"] and r0["residual"] == r1["residual"] == 0
    refs = B._refs(g)
    assert refs[4] == 2 and refs[5] == 2           # the ancestor and the locked node
    assert refs[6] == 0 and refs[7] == 0           # below the locked node: leaves with copies leave
    for t, nodes in ((g.t0, g.n0), (g.t1, g.n1)):
        alive = _alive(t)
        assert nodes[6] not in alive and nodes[7] not in alive and nodes[5] in alive


def test_tp0_tells_the_workers_every_node_it_removed(rc27):
    """R12: TP0's removal of a host-only node is a STATE(exists=0) for its shadow of the workers (the K23 form)."""
    g = rc27
    B._sleep_release(g)
    sent = {(e[1], int(e[2])) for e in R._S.ledger if e[0] == R.STATE and e[3] == 0}
    assert {R.node_key(n) for n in g.n0[DEV:]} <= sent, sent


def test_every_rank_posts_the_same_collectives(rc27):
    g = rc27
    (_r0, _r1), grp = B._sleep_release(g)
    assert grp.calls[0] == grp.calls[1] == [
        "s86_sleep_l3_release/slots", "s86_sleep_l3_release/veto", "s86_sleep_l3_release/veto_fix",
        "s86_sleep_l3_release/digest"]


# ---- (3) the aux rule (veto level): only a MAMBA anchor, judged by the rank that owns it -------------------------

class _MambaCt:
    pass


def _aux_tree(tmp_path, owner_pool=True):
    """One rank, FULL + MAMBA components; node 0 device-resident with a host-only mamba anchor, node 1 host-only with
    a host-only mamba anchor (a leaf under node 0)."""
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    R._reset_state_for_tests()
    g = B._group(tmp_path)
    t, nodes = g.t0, g.n0
    for n in nodes:
        assert g.p0.secure_rows_to_l3(n.component_data[FULL].host_value)["lost"] == 0
    t._components_tuple = (t.components[FULL], types.SimpleNamespace(component_type=ComponentType.MAMBA))
    nodes[0].component_data[ComponentType.MAMBA].host_value = torch.tensor([3], dtype=torch.int64)
    nodes[-1].component_data[FULL].value = None
    nodes[-1].component_data[ComponentType.MAMBA].host_value = torch.tensor([4], dtype=torch.int64)
    t._pdflip_mamba_pool = (lambda: types.SimpleNamespace(arena=object())) if owner_pool else (lambda: None)
    return g, t, nodes


def _veto(t, pool):
    return S86.local_veto(t, pool, SLOTS)


def test_aux_mamba_anchor_secured_makes_both_nodes_candidates(tmp_path):
    from flliper.srt.pdflip import forma_l3_subtree as FLS

    g, t, nodes = _aux_tree(tmp_path)
    with mock.patch.object(FLS, "_secure_child_anchors", lambda tree, order, budget: None):
        veto, cands, why = _veto(t, g.p0)
    names = {c[0] for c in cands}
    assert nodes[0] in names and nodes[-1] in names, why
    assert int(veto.sum()) == 0


def test_aux_mamba_anchor_pending_keeps_the_node_named(tmp_path):
    """Danger edge: an anchor the write-behind has not copied keeps its node (``anchor_pending``) -- on the device
    node AND on the host-only leaf."""
    from flliper.srt.pdflip import forma_l3_subtree as FLS

    g, t, nodes = _aux_tree(tmp_path)
    with mock.patch.object(FLS, "_secure_child_anchors",
                           lambda tree, order, budget: ("child_anchor_pending", order[0].id)):
        veto, cands, why = _veto(t, g.p0)
    names = {c[0] for c in cands}
    assert nodes[0] not in names and nodes[-1] not in names
    assert why.get("anchor_pending") == 1 and why.get("host_only:anchor_pending") == 1, why


def test_aux_on_a_rank_without_an_anchor_pool_is_not_judged_there(tmp_path):
    """A Form A worker owns no anchor pool: the anchor is TP0's to judge (its veto reaches the worker through the
    group MAX). The anchor helper is not asked on the worker."""
    from flliper.srt.pdflip import forma_l3_subtree as FLS

    g, t, nodes = _aux_tree(tmp_path, owner_pool=False)
    asked = []
    with mock.patch.object(FLS, "_secure_child_anchors", lambda *a: asked.append(1) or ("child_anchor_pending", 0)):
        veto, cands, why = _veto(t, g.p0)
    assert asked == []
    assert {nodes[0], nodes[-1]} <= {c[0] for c in cands}


def test_another_aux_type_keeps_the_old_veto(tmp_path):
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    swa = getattr(ComponentType, "SWA", None)
    if swa is None:
        pytest.skip("no SWA component type in this tree")
    g, t, nodes = _aux_tree(tmp_path)
    t._components_tuple = t._components_tuple + (types.SimpleNamespace(component_type=swa),)
    nodes[0].component_data[swa].host_value = torch.tensor([1], dtype=torch.int64)
    veto, cands, why = _veto(t, g.p0)
    assert nodes[0] not in {c[0] for c in cands}
    assert why.get("aux_host_only") == 1, why


def test_a_host_only_node_whose_rows_name_no_arena_page_stays(tmp_path):
    """Danger edge: host rows outside the arena have no L3 proof -- never removed (``host_only:no_arena_slot``)."""
    g, t, nodes = _aux_tree(tmp_path)
    with mock.patch.object(S86, "_slots", lambda pool, hv: []):
        veto, cands, why = _veto(t, g.p0)
    assert nodes[-1] not in {c[0] for c in cands}
    assert why.get("host_only:no_arena_slot") == 1, why


def test_s86_unbacked_census_still_names_what_the_reset_drops(rc27):
    """What the reset still drops (no copy anywhere) is named with count and tokens: with the host-only room gone
    too (every node host-locked on the worker), the S85 census names the chain."""
    g = rc27
    for n in g.n1:
        n.component_data[FULL].host_lock_ref = 1
    B._sleep_release(g)
    B._flush_round(g)
    ents = F.note_unbacked_at_reset(g.t0, site="flush").get("park_end") or []
    assert sum(int(e["lost_tokens"]) for e in ents) == 2


# ---- (4) three ranks, rank-local host-only trees (the S86c shape, host-only) ------------------------------------

def _three(tmp_path, lock=None):
    """TP0 holds a0..a5 (a4, a5 host-only), TP1 and TP2 hold a0..a7 (a4..a7 host-only); TP2's last node spans a6+a7
    (a rank-local split). Every page in L3. ``lock`` = (rank, index) host-locks one node."""
    import threading  # noqa: F401

    R._reset_state_for_tests()
    store = tmp_path / "store"
    store.mkdir()
    ap = str(tmp_path / "kv.bin")
    pools, arenas = [], []
    for _ in range(3):
        p, a = D2._rank_pool(tmp_path, ap, str(store))
        pools.append(p)
        arenas.append(a)
    hashes = [[f"a{i}"] for i in range(SLOTS)]
    pay = torch.zeros((W.PAGE,), dtype=torch.uint8)
    for i, hs in enumerate(hashes):
        pay.fill_(0x40 + i)
        assert arenas[0].write(pools[0]._stems(hs), [W.PAGE], [((0, W.PAGE),)], [pay.data_ptr()]) == [1]
    trees = [D2._tree(p) for p in pools]
    nodes = [D2._mirror(trees[0], pools[0], arenas[0], hashes[:6]),
             D2._mirror(trees[1], pools[1], arenas[1], hashes),
             D2._mirror(trees[2], pools[2], arenas[2], hashes)]
    for n in nodes[1]:
        assert pools[1].secure_rows_to_l3(n.component_data[FULL].host_value)["lost"] == 0
    n6, n7 = nodes[2][6], nodes[2][7]
    n6.component_data[FULL].host_value = torch.cat([n6.component_data[FULL].host_value,
                                                    n7.component_data[FULL].host_value])
    n6.hash_value = ["a6", "a7"]
    n6.children.clear()
    nodes[2] = nodes[2][:7]
    for t, nd in zip(trees, nodes):
        _host_leaf_removal(t)
        for n in nd[DEV:]:
            n.component_data[FULL].value = None
    if lock is not None:
        nodes[lock[0]][lock[1]].component_data[FULL].host_lock_ref = 1
    return types.SimpleNamespace(pools=pools, arenas=arenas, trees=trees, nodes=nodes)


def _run3(g):
    import threading

    grp = B._Group(3)
    res = [None] * 3
    roles = ("host", "worker", "worker")

    def one(i):
        B._ROLE.r = roles[i]
        try:
            res[i] = S86.run(g.trees[i], grp.member(i), g.pools[i])
        except BaseException as e:  # noqa: BLE001
            res[i] = e
        finally:
            B._ROLE.r = None

    ths = [threading.Thread(target=one, args=(i,)) for i in range(3)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    return res, grp


def test_three_ranks_with_rank_local_host_only_trees_release_one_set(tmp_path):
    g = _three(tmp_path)
    res, grp = _run3(g)
    assert not any(isinstance(r, BaseException) for r in res), res
    assert grp.calls[0] == grp.calls[1] == grp.calls[2]
    assert all(r["agreed"] and r["residual"] == 0 for r in res), res
    refs = [int(r) for r in g.arenas[0].slot_refs(list(range(SLOTS)))]
    assert refs == [0] * SLOTS, refs        # device half (plain) and host-only half both released, every rank


def test_three_ranks_a_lock_inside_the_split_node_keeps_both_halves_everywhere(tmp_path):
    """TP1's a7 is host-locked; TP2's node spans a6+a7, so a6 is kept on TP2 (mixed node) and then on every rank
    (the fixed point), and with them the host-only ancestors a4, a5 (subtree rule). One agreed set, residual 0."""
    g = _three(tmp_path, lock=(1, 7))
    res, _grp = _run3(g)
    assert not any(isinstance(r, BaseException) for r in res), res
    assert all(r["agreed"] and r["residual"] == 0 for r in res), res
    refs = [int(r) for r in g.arenas[0].slot_refs(list(range(SLOTS)))]
    assert refs[:DEV] == [0] * DEV, refs
    assert all(r > 0 for r in refs[DEV:]), refs


def test_a_child_vetoed_only_by_a_peer_that_holds_the_parents_on_the_device_keeps_the_parents(rc27):
    """The fixed point carries the subtree rule ACROSS node kinds: TP0's a4..a7 are host-only; the worker holds a4..a6
    on the DEVICE (released by the device rule) and a7 host-only but read by a load. Only a7 is vetoed locally; TP0's
    a6 cannot leave without a7, so a6 joins the veto, then the worker keeps a6's host copy, then a5, a4. One agreed
    set, residual 0 on both ranks (a release of a4..a6 on the worker alone would be a W86 divergence)."""
    g = rc27
    for n in g.n1[DEV:7]:
        n.component_data[FULL].value = torch.arange(1, dtype=torch.int64)   # on the worker's device
    g.n1[7].component_data[FULL].host_lock_ref = 1
    (r0, r1), _grp = B._sleep_release(g)
    assert r0["agreed"] and r1["agreed"] and r0["residual"] == r1["residual"] == 0, (r0, r1)
    refs = B._refs(g)
    assert refs[DEV:] == [2] * (SLOTS - DEV), refs
    assert all(n in _alive(g.t0) for n in g.n0[DEV:])
