"""S86c (rc23 boot ..._d76cae6851_1020_105621, D 11:05:06Z, group death):

    S86 SLEEP-L3-RELEASE ... TP0 nodes=33 slots=431 kept=aux_host_only:24,host_only:8
                             TP1/TP2 nodes=35 slots=433 kept=host_only:9,aux_host_only:8
    W86 SleepL3ReleaseDivergence: ... DIVERGED -- released 431..433 slots ... -> W29 PdFlipRankDisagree, group dead
    (ARENA-REF-HOLDERS just before: TP0 tree=259, TP2 tree=433)

The node structure is RANK-LOCAL on a Form A D group: TP0 (share-0 host) holds other nodes than the token owners,
and splits are rank-local. Two inputs of the release were rank-local: (1) a node holding an agreed slot AND a
vetoed one is kept whole on its rank while a peer lets the agreed slot go -- a real divergence; (2) a rank that does
not hold a slot at all releases a smaller COUNT -- no divergence, yet the old digest compared the counts.

The fix: a group fixed point over the veto (mixed nodes block their agreed slots, MAX until the reduced vector
stands) and a unity check on what must be one -- the agreed set's digest and every rank's RESIDUAL (agreed slots it
holds but kept), not the released count.

THREE ranks on one real C arena (real ArenaMHAHostPool each, real secure_rows_to_l3, threads, MAX behind a barrier):
TP0 holds no node for a6/a7 (the metal's smaller TP0 tree), TP1's node spans a2+a3 with a3 host-locked, TP2 plain.
RED on f54d033012 (W86 on every rank -- the rc23 death); the mutant "rank-local selection" (no fixed point) is red."""
import os
import sys
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_s86_sleep_l3_release_1020 as B  # noqa: E402
import test_w3_arena_spill_0929 as W  # noqa: E402
import test_w3_arena_spill_forma_dual15_0930 as D2  # noqa: E402
from test_s86_sleep_l3_release_1020 import _thread_roles  # noqa: E402,F401

from flliper.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from flliper.srt.pdflip import sleep_l3_release as S86  # noqa: E402

FULL, SLOTS = W.FULL, 8
ROLES = ("host", "worker", "worker")


@pytest.fixture
def three(tmp_path):
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
    nodes = [D2._mirror(trees[0], pools[0], arenas[0], hashes[:6]),       # TP0: no node for a6, a7
             D2._mirror(trees[1], pools[1], arenas[1], hashes),
             D2._mirror(trees[2], pools[2], arenas[2], hashes)]
    for n in nodes[1]:                                                     # every page in L3 (as on the metal)
        assert pools[1].secure_rows_to_l3(n.component_data[FULL].host_value)["lost"] == 0
    # TP1: its node 2 spans a2 AND a3 (rank-local split), and a3 is read by a load (host lock)
    n2 = nodes[1][2]
    n2.component_data[FULL].host_value = torch.cat([n2.component_data[FULL].host_value,
                                                    nodes[1][3].component_data[FULL].host_value])
    n2.hash_value = ["a2", "a3"]
    nodes[1][3].component_data[FULL].host_lock_ref = 1
    return types.SimpleNamespace(pools=pools, arenas=arenas, trees=trees, nodes=nodes)


def _run3(g):
    grp = B._Group(3)
    res = [None] * 3

    def one(i):
        B._ROLE.r = ROLES[i]
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


def _refs(g):
    return [int(r) for r in g.arenas[0].slot_refs(list(range(SLOTS)))]


def test_rc23_three_ranks_with_their_own_trees_release_one_set(three):
    g = three
    assert _refs(g) == [3, 3, 3, 3, 3, 3, 2, 2]          # TP0 holds no a6/a7 (its smaller tree)
    res, grp = _run3(g)
    assert not any(isinstance(r, BaseException) for r in res), res
    assert grp.calls[0] == grp.calls[1] == grp.calls[2]
    # one set: a2/a3 kept by every holder (the spanning node cannot let a2 go), everything else freed
    refs = _refs(g)
    assert [refs[i] for i in (0, 1, 4, 5, 6, 7)] == [0] * 6, refs
    assert refs[2] == 3 and refs[3] == 3, refs
    for i in (2, 3):
        assert all(nd[i].component_data[FULL].host_value is not None for nd in g.nodes)
    # the counts differ (TP0 holds no a6/a7), which is no divergence: residual 0 everywhere, agreed
    assert [r["pages"] for r in res] == [4, 6, 6]
    assert all(r["residual"] == 0 and r["agreed"] for r in res)


def test_rc23_the_released_count_alone_differs_and_is_not_a_divergence(three):
    """Without the mixed node: TP0 still holds no a6/a7 -- 6 released on TP0, 8 on the workers, one agreed set."""
    g = three
    n2 = g.nodes[1][2]
    n2.component_data[FULL].host_value = n2.component_data[FULL].host_value[:1]
    n2.hash_value = ["a2"]
    g.nodes[1][3].component_data[FULL].host_lock_ref = 0
    res, grp = _run3(g)
    assert not any(isinstance(r, BaseException) for r in res), res
    assert [r["pages"] for r in res] == [6, 8, 8]
    assert _refs(g)[:2] == [0, 0] and _refs(g)[4:] == [0, 0, 0, 0]
