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
