"""F1 (pgap root, h16s boot 1016_103428): ONE slot set for the W3 spill of a PP group.

Metal: P's KV arena 6485/6485 pinned by the three stages' own host copies; each stage picked its own
candidates (its trees are not replicas), so of 214-227 pages released per spill only 13-15 slots
became free. Here: three stages on ONE real C arena file (the dual15 harness, three pools), every
stage holds pages a0..a3 (refs 3); PP1 also holds a private page c0 (refs 1), so by the group-equal
order PP1 would spill c0, not a0 -- a0 keeps PP1's reference and never frees.
F1: PP0 decides {slots} and publishes them next to the arena; every stage gives back exactly those
(L3 copy first, COMPLETE only) -> each slot of the set reaches 0 references.
RED on 9095d19351 (h16s + A1 + K15c): a0 stays referenced after all three stages spilled."""
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_w3_arena_spill_0929 as W  # noqa: E402
import test_w3_arena_spill_forma_dual15_0930 as F  # noqa: E402

from sglang.srt.weg2 import w3_room_set as RS  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
FULL = F.FULL


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(RS.ENV, raising=False)


@pytest.fixture
def pp3(tmp_path):
    from sglang.srt.mem_cache import form_a_host_shadow as R

    R._reset_state_for_tests()
    store = tmp_path / "store"
    store.mkdir()
    ap = str(tmp_path / "kv.bin")
    pools = [F._rank_pool(tmp_path, ap, str(store)) for _ in range(3)]
    shared = [[f"a{i}"] for i in range(4)]
    pay = torch.zeros((W.PAGE,), dtype=torch.uint8)
    p0, a0 = pools[0]
    for i, hs in enumerate(shared + [["c0"]]):
        pay.fill_(0x40 + i)
        assert a0.write(p0._stems(hs), [W.PAGE], [((0, W.PAGE),)], [pay.data_ptr()]) == [1]
    trees, nodes = [], []
    for r, (p, a) in enumerate(pools):
        t = F._tree(p)
        t.pp_size, t.pp_rank = 3, r
        t.W3_SPILL_MIN_PAGES = 1
        t._w3_spill_quiet_until = 0.0
        t._w3_keep_payable_state = lambda: None
        layout = {0: shared, 1: [["c0"]] + shared, 2: [shared[0] + shared[1], shared[2] + shared[3]]}[r]
        # stage-local node boundaries differ (PP2 holds two-page nodes)
        ns = []
        for hs in layout:
            ns += F._mirror(t, p, a, [hs])
        trees.append(t)
        nodes.append(ns)
    return pools, trees, nodes, store


def _all_nodes(tree):
    return [n for n in tree._collect_all_nodes() if n is not tree.root_node]


def _refs(arena, p, hashes):
    slots = [int(s) for s, _ in arena.find_slots(p._stems(hashes))]
    return arena.slot_refs(slots)


def test_pp_group_frees_the_slot_pp0_chose(pp3):
    pools, trees, nodes, store = pp3
    (p0, a0) = pools[0]
    for t, (p, a) in zip(trees, pools):         # PP0 first (it decides), then the followers
        t._w3_arena_spill(p, 1)
    gen, slots = RS.read(a0)
    assert gen >= 1 and slots, "PP0 published no slot set"
    a0_slot = int(a0.find_slots(p0._stems(["a0"]))[0][0])
    assert a0_slot in slots
    assert _refs(a0, p0, ["a0"]) == [0], "the chosen slot keeps a stage's reference"
    assert _refs(a0, p0, ["c0"]) == [1], "PP1 spilled a page outside PP0's set"


def test_follower_applies_at_its_next_claim_without_its_own_refusal(pp3):
    pools, trees, nodes, store = pp3
    (p0, a0), (p1, a1) = pools[0], pools[1]
    trees[0]._w3_arena_spill(p0, 1)
    r = RS.apply_if_new(trees[1], p1)
    assert r is not None and r["nodes"] >= 1
    assert RS.apply_if_new(trees[1], p1) is None, "one apply per generation"


def test_running_request_node_is_never_a_candidate(pp3):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    first = nodes[0][0]
    first.component_data[FULL].lock_ref = 1      # a running request holds a0 on PP0
    slots = RS.decide(trees[0], p0, 1)
    a0_slot = int(a0.find_slots(p0._stems(["a0"]))[0][0])
    assert a0_slot not in slots


def test_switch_off_is_the_per_rank_choice(pp3, monkeypatch):
    monkeypatch.setenv(RS.ENV, "0")
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    for t, (p, a) in zip(trees, pools):
        t._w3_arena_spill(p, 1)
    assert RS.read(a0)[0] == 0
    assert _refs(a0, p0, ["a0"]) != [0]           # the metal's divergence: a0 not freed


# --- review F1 (NEIN, 16.10.): A1-A6, probes P1-P6 of /spinning/wt-rev-f1-1016 -------------------------

def _slot(a, p, h):
    return int(a.find_slots(p._stems([h]))[0][0])


def _round(trees, pools, need=1):
    for t, (p, a) in zip(trees, pools):
        t._w3_spill_quiet_until = 0.0
        t._w3_arena_spill(p, need)


def test_review_p1_a_freed_and_adopted_page_is_not_evicted_again(pp3):
    """A1: gen N freed a0 on all stages; a new request adopts a0 again (refs 3) -- gen N+1 must not
    carry a0's slot (refs > 0 cannot tell a laggard from a re-adoption)."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    _round(trees, pools)
    s_a0 = _slot(a0, p0, "a0")
    assert _refs(a0, p0, ["a0"]) == [0]
    for t, (p, a) in zip(trees, pools):
        F._mirror(t, p, a, [["a0"]])
    assert _refs(a0, p0, ["a0"]) == [3]
    _round(trees, pools)
    gen, slots = RS.read(a0)
    assert gen >= 2 and s_a0 not in slots, (gen, slots)
    assert _refs(a0, p0, ["a0"]) == [3]


def test_review_a1_the_set_does_not_grow_under_full_readoption(pp3):
    """A1: N generations, every freed page adopted again at once -- each set stays one decision wide."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    sizes, gens = [], []
    for _ in range(5):
        _round(trees, pools)
        gen, slots = RS.read(a0)
        sizes.append(len(slots))
        gens.append(gen)
        for h in ("a0", "a1", "a2", "a3"):
            if _refs(a0, p0, [h]) == [0]:
                for t, (p, a) in zip(trees, pools):
                    F._mirror(t, p, a, [[h]])
    assert max(sizes) <= 1 and gens == sorted(set(gens)), (sizes, gens)


def test_review_p3_an_earlier_boots_set_is_never_applied(pp3):
    """A5: an old-format file, and a current-format file written before this process started."""
    import struct

    pools, trees, nodes, store = pp3
    (p0, a0), (p1, a1) = pools[0], pools[1]
    s_a2 = _slot(a0, p0, "a2")
    path = str(a0.path) + RS.SUFFIX
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", 77, 1))
        f.write(struct.pack("<q", s_a2))
    assert RS.apply_if_new(trees[1], p1) is None
    RS.publish(a0, [s_a2], trees[0])
    os.utime(path, ns=(RS._PROC_T0_NS - 10**9, RS._PROC_T0_NS - 10**9))
    assert RS.read(a0) == (0, [])
    assert RS.apply_if_new(trees[1], p1) is None
    assert _refs(a0, p0, ["a2"]) == [3]


def test_review_p4_a_slot_only_a_follower_holds_is_freed_by_it(pp3):
    """A3: PP0 has no candidate; PP1 holds c0 alone -> PP1 releases it. A slot PP0/PP2 reference too is
    never PP1's choice (the mutant 'every stage decides' stays red)."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    for n in nodes[0]:
        n.component_data[FULL].lock_ref = 1
    _round(trees, pools)
    assert _refs(a0, p0, ["c0"]) == [0]
    assert [_refs(a0, p0, [h])[0] for h in ("a0", "a1", "a2", "a3")] == [3, 3, 3, 3]


def test_review_p5_keep_payable_holds_under_f1(pp3, monkeypatch):
    """A2: W3-KEEP-PAYABLE (ARENAFULL-1015): a node whose child is not removable from the host keeps its
    host copy under F1 too."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode

    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    a0n = nodes[0][0]
    z = UnifiedTreeNode((FULL,))
    z.parent = a0n
    a0n.children[("z",)] = z
    z.component_data[FULL].value = torch.arange(1)
    z.component_data[FULL].lock_ref = 1
    state = {id(z): (0, False, False)}
    monkeypatch.setattr(UnifiedRadixCache, "_w3_keep_payable_state", lambda self: state)
    trees[0]._w3_arena_spill(p0, 1)
    assert a0n.component_data[FULL].host_value is not None
    assert _slot(a0, p0, "a0") not in RS.read(a0)[1]


def test_review_a2_the_claimers_chain_is_not_chosen(pp3):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    claimer = nodes[0][0]
    slots = RS.decide(trees[0], p0, 1, claimer=claimer)
    assert _slot(a0, p0, "a0") not in slots and slots


def test_review_p6_followers_first_then_one_tick_frees_the_slot(pp3, monkeypatch):
    """A4: READ-ROOM serve order PP1, PP2, PP0, then one follower round: the slot is free, and no
    follower sits in the empty backoff."""
    from sglang.srt.weg2 import read_room as RR

    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    monkeypatch.setattr(RR, "deficit", lambda arena, need: (int(need), {"slots": 8, "complete": 0, "claimed": 0,
                                                                        "pinned": 0, "free": 0, "clock": 0}))
    for t in trees:
        t._w3_spill_quiet_until = 0.0
    for r in (1, 2, 0):
        RR.serve(trees[r], pools[r][0], pools[r][1], 1, read_key=("x", 1))
    for r in (1, 2):
        assert RS.apply_if_new(trees[r], pools[r][0]) is not None
    gen, slots = RS.read(a0)
    assert slots and all(int(x) == 0 for x in a0.slot_refs(slots)), a0.slot_refs(slots)


def test_review_a4_pp0_decides_no_new_set_before_the_followers_acknowledge(pp3):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    trees[0]._w3_arena_spill(p0, 1)
    g1 = RS.read(a0)[0]
    trees[0]._w3_spill_quiet_until = 0.0
    assert trees[0]._w3_arena_spill(p0, 1) == 0
    assert RS.read(a0)[0] == g1 and RS.undecided(trees[0])
    assert trees[0]._w3_spill_quiet_until == 0.0, "an undecided spill set the empty backoff"
    for r in (1, 2):
        RS.apply_if_new(trees[r], pools[r][0])
    trees[0]._w3_arena_spill(p0, 1)
    assert RS.read(a0)[0] == g1 + 1


def test_review_p2_a_follower_debt_is_released_once_the_request_ends(pp3):
    """The follower's node is locked when PP0's set arrives: it goes on the debt list; once the
    request ends, the next round releases it (no new generation needed)."""
    pools, trees, nodes, store = pp3
    (p0, a0), (p1, a1) = pools[0], pools[1]
    n_a0 = [n for n in nodes[1] if n.key == ["a0"]][0]
    n_a0.component_data[FULL].lock_ref = 1
    trees[0]._w3_arena_spill(p0, 1)
    assert _slot(a0, p0, "a0") in RS.read(a0)[1]
    RS.apply_if_new(trees[1], p1)
    RS.apply_if_new(trees[2], pools[2][0])
    assert _refs(a0, p0, ["a0"]) == [1]
    n_a0.component_data[FULL].lock_ref = 0
    r = RS.apply_if_new(trees[1], p1)
    assert r is not None and r["nodes"] == 1
    assert _refs(a0, p0, ["a0"]) == [0]


def test_review_a6_the_page_cap_bounds_one_decision(pp3, monkeypatch):
    monkeypatch.setenv(RS.CAP_ENV, "1")
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    assert len(RS.decide(trees[0], p0, 4)) == 1


def test_review_a4_wiring_every_read_room_round_ticks_the_room_set(monkeypatch):
    from sglang.srt.weg2 import read_room as RR

    seen = []
    monkeypatch.setattr(RS, "tick", lambda cache: seen.append(cache))
    monkeypatch.setattr(RR, "armed", lambda cache: False)
    c = object()
    RR.tick(c)
    assert seen == [c]
