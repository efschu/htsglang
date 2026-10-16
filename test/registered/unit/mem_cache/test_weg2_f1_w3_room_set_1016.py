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
    monkeypatch.setattr(_rr(), "deficit", lambda arena, need: (int(need), {"slots": 8, "complete": 0, "claimed": 0,
                                                                        "pinned": 0, "free": 0, "clock": 0}))
    for t in trees:
        t._w3_spill_quiet_until = 0.0
    for r in (1, 2, 0):
        RR.serve(trees[r], pools[r][0], pools[r][1], 1, read_key=("x", 1))
    assert not any(t._w3_spill_quiet_until > 0 for t in trees[1:]), "a follower sat in the empty backoff (B6)"
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


# --- F1b review (JA MIT AUFLAGE, 16.10.): B1-B6, reviewer probes Q1-Q7 of /spinning/wt-rev-f1b-1016 ----------

RR = None


def _rr():
    from sglang.srt.weg2 import read_room
    return read_room


def test_review_f1b_q1_a_used_up_cap_is_no_cap_off(pp3, monkeypatch):
    """CAP 1: PP0 applies a set of 3 single-page nodes; the cap must hold at 1 page per call."""
    monkeypatch.setenv(RS.CAP_ENV, "1")
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    s = {_slot(a0, p0, h) for h in ("a1", "a2", "a3")}
    # a1..a3 on PP0 are a chain a0->a1->a2->a3; a3 is a host leaf? use a device-node release instead
    r = RS.apply(trees[0], p0, s)
    print("apply", r)
    assert r["nodes"] <= 1, "cap_left reached exactly 0 and then the cap no longer applied: %r" % (r,)


def test_review_f1b_q2_a_raising_apply_still_acks_and_pp0_decides_again(pp3, monkeypatch):
    """A follower whose apply raises (deterministically) never acknowledges; PP0 then never decides again."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    trees[0]._w3_arena_spill(p0, 1)
    g1 = RS.read(a0)[0]
    p1 = pools[1][0]

    def boom(hv):
        raise OSError("L3 write failed")
    monkeypatch.setattr(p1, "secure_rows_to_l3", boom)
    for _ in range(3):
        try:
            RS.apply_if_new(trees[1], p1)
        except OSError:
            pass
    RS.apply_if_new(trees[2], pools[2][0])
    gens = []
    for _ in range(3):
        trees[0]._w3_spill_quiet_until = 0.0
        trees[0]._w3_arena_spill(p0, 1)
        gens.append(RS.read(a0)[0])
    print("g1", g1, "gens", gens, "undecided", RS.undecided(trees[0]))
    assert gens[-1] > g1, "PP0 frozen: a follower that cannot apply never acks, no timeout"


def test_review_f1b_q3_a_debt_on_a_readopted_slot_is_dropped(pp3):
    """PP1's a0 is locked when PP0's set arrives (debt). PP0 then adopts a0 again (new request).
    When PP1's request ends, the debt still releases PP1's host copy although the slot gains no room."""
    pools, trees, nodes, store = pp3
    (p0, a0), (p1, a1) = pools[0], pools[1]
    n_a0 = [n for n in nodes[1] if n.key == ["a0"]][0]
    n_a0.component_data[FULL].lock_ref = 1
    trees[0]._w3_arena_spill(p0, 1)
    assert _slot(a0, p0, "a0") in RS.read(a0)[1]
    RS.apply_if_new(trees[1], p1)
    RS.apply_if_new(trees[2], pools[2][0])
    assert _refs(a0, p0, ["a0"]) == [1]
    F._mirror(trees[0], p0, a0, [["a0"]])           # PP0 re-adopts a0 (fresh request)
    assert _refs(a0, p0, ["a0"]) == [2]
    n_a0.component_data[FULL].lock_ref = 0
    r = RS.apply_if_new(trees[1], p1)
    print("r", r, "refs", _refs(a0, p0, ["a0"]), "pp1 host", n_a0.component_data[FULL].host_value is not None)
    assert n_a0.component_data[FULL].host_value is not None, \
        "debt released PP1's host copy of a slot PP0 had re-adopted (no room gained)"


def test_review_f1b_q4_an_idle_round_walks_nothing(pp3, monkeypatch):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    pools, trees, nodes, store = pp3
    calls = []
    monkeypatch.setattr(UnifiedRadixCache, "_w3_keep_payable_state", lambda self: calls.append(1) or None)
    for r in (0, 1, 2):
        for _ in range(5):
            RS.apply_if_new(trees[r], pools[r][0])
    print("keep-payable walks with no set and no debts:", len(calls))
    assert len(calls) == 0, "every 20 ms tick builds _Ctx (full payable walk) even with nothing to do"


def test_review_f1b_q5_a_host_locked_debt_is_decided(pp3, monkeypatch):
    """PP1's a0 is host-locked (told span of the waiting read). After PP0's set, PP1's spill is
    UNDECIDED for as long as the lock holds -> F2 can never call the wait futile."""
    pools, trees, nodes, store = pp3
    (p0, a0), (p1, a1) = pools[0], pools[1]
    for n in nodes[1]:
        n.component_data[FULL].host_lock_ref = 1
    trees[0]._w3_arena_spill(p0, 1)
    out = []
    for _ in range(4):
        trees[1]._w3_spill_quiet_until = 0.0
        rel = trees[1]._w3_arena_spill(p1, 1)
        out.append((rel, RS.undecided(trees[1])))
    print("pp1 spills", out, "debts", len(trees[1]._w3_room_debt))
    assert not out[-1][1], "a debt blocked by the read's own host lock keeps the follower UNDECIDED"


def test_review_f1b_q6_followers_first_then_tick_no_backoff(pp3, monkeypatch):
    """Original probe P6 + one follower READ-ROOM tick (RS.tick path with the 20 ms gate)."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    monkeypatch.setattr(_rr(), "deficit", lambda arena, need: (int(need), {"slots": 8, "complete": 0, "claimed": 0,
                                                                        "pinned": 0, "free": 0, "clock": 0}))
    for t in trees:
        t._w3_spill_quiet_until = 0.0
    rel = {}
    for r in (1, 2, 0):
        rel[r] = _rr().serve(trees[r], pools[r][0], pools[r][1], 1, read_key=("x", 1))
    quiet = [t._w3_spill_quiet_until > 0 for t in trees]
    for r in (1, 2):
        trees[r]._weg2_direct_pool = (lambda p: (lambda: p))(pools[r][0])
        trees[r]._w3_room_tick_t = 0.0
        RS.tick(trees[r])
    gen, slots = RS.read(a0)
    print("rel", rel, "quiet", quiet, "gen", gen, "slots", slots, "refs", a0.slot_refs(slots))
    assert slots and all(int(x) == 0 for x in a0.slot_refs(slots))
    assert not any(quiet[1:]), "a follower sat in the empty backoff"


def test_review_f1b_q7_a_reset_trees_debt_never_frees_twice(pp3):
    """PP1's a0 is a debt (locked). Then the tree is reset (flip / P sleep flush): _reset_full gives the old
    tree's arena references back (release_tree_rows, host_value left set) and swaps the root, but it does not
    clear _w3_room_debt. PP1 adopts a0 again under the NEW root. The next 20 ms tick pays the stale debt:
    the dead node's host rows are freed a second time -> the new holder's reference is taken off."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
    pools, trees, nodes, store = pp3
    (p0, a0), (p1, a1) = pools[0], pools[1]
    t1 = trees[1]
    n_a0 = [n for n in nodes[1] if n.key == ["a0"]][0]
    n_a0.component_data[FULL].lock_ref = 1
    trees[0]._w3_arena_spill(p0, 1)
    RS.apply_if_new(t1, p1)
    RS.apply_if_new(trees[2], pools[2][0])
    assert id(n_a0) in t1._w3_room_debt
    # --- reset as _reset_full does it: refs back (host_value stays), new root, sets/write-throughs fresh
    stack = list(t1.root_node.children.values())
    while stack:
        n = stack.pop()
        stack.extend(n.children.values())
        hv = n.component_data[FULL].host_value
        if hv is not None:
            p1.release_tree_rows(hv)
    new_root = UnifiedTreeNode((FULL,))
    new_root.component_data[FULL].lock_ref = 1
    new_root.component_data[FULL].value = []
    t1.root_node = new_root
    t1.evictable_host_leaves = set()
    t1.ongoing_write_through = {}
    n_a0.component_data[FULL].lock_ref = 0          # the dead tree's request ended
    # PP1 adopts a0 again under the new tree
    fresh = F._mirror(t1, p1, a1, [["a0"]])
    before = _refs(a0, p0, ["a0"])
    r = RS.apply_if_new(t1, p1)
    after = _refs(a0, p0, ["a0"])
    print("before", before, "apply", r, "after", after, "debts", len(t1._w3_room_debt))
    assert after == before, "a stale debt from the reset tree released the NEW holder's reference"



def test_review_f1b_b1_wiring_the_tree_reset_forgets_the_debts():
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    assert "forget(self)" in inspect.getsource(UnifiedRadixCache._reset_full)


def test_review_f1b_b1_forget_clears_the_debt_list(pp3):
    pools, trees, nodes, store = pp3
    t1 = trees[1]
    t1._w3_room_debt = {1: (object(), frozenset({1}), 1, "running")}
    RS.forget(t1)
    assert not RS._debts(t1)


def test_review_f1b_b3_pp0_decides_anew_after_the_wait_bound(pp3, monkeypatch):
    """A follower never acknowledges (dead round, stuck apply): PP0 decides again after WAIT_TIMEOUT_S."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    trees[0]._w3_arena_spill(p0, 1)
    g1 = RS.read(a0)[0]
    trees[0]._w3_spill_quiet_until = 0.0
    trees[0]._w3_arena_spill(p0, 1)
    assert RS.read(a0)[0] == g1                    # inside the bound: waits
    monkeypatch.setattr(RS, "WAIT_TIMEOUT_S", 0.0)
    trees[0]._w3_spill_quiet_until = 0.0
    trees[0]._w3_arena_spill(p0, 1)
    assert RS.read(a0)[0] == g1 + 1
