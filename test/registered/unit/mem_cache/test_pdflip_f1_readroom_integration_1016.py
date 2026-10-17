"""F1 review A8 (16.10.): READ-ROOM x F1 on ONE real C arena with three trees (pp_size=3) -- the real
``read_room.serve`` -> ``_w3_arena_spill`` -> ``w3_room_set`` path, followers served FIRST, a page freed
and adopted again, a slot only a follower holds, and the room-verdict flag (``undecided``) a F2 serve
writes as 'did not look' (merge line on the pgap branch: ``_rv.note_verdict(arena, released,
_rv.transient_blocked(cache), _quiet or _w3rs.undecided(cache))``).
The header deficit is pinned (the fixture arena is not full); everything else is the production path."""
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_pdflip_f1_w3_room_set_1016 as T  # noqa: E402

from flliper.srt.pdflip import read_room as RR  # noqa: E402
from flliper.srt.pdflip import w3_room_set as RS  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
pp3 = T.pp3
_env = T._env
FULL = T.FULL


def _serve_round(trees, pools, order, key):
    out = {}
    for r in order:
        trees[r]._w3_spill_quiet_until = 0.0
        out[r] = RR.serve(trees[r], pools[r][0], pools[r][1], 1, read_key=key)
    return out


def test_a8_readroom_f1_followers_first_readoption_and_follower_only_slot(pp3, monkeypatch):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    monkeypatch.setattr(RR, "deficit", lambda arena, need: (int(need), {
        "slots": 8, "complete": 0, "claimed": 0, "pinned": 0, "free": 0, "clock": 0}))
    # 1) followers serve before PP0 decides: nothing to apply, no empty answer pending a decision
    rel = _serve_round(trees, pools, (1, 2, 0), ("r", 1))
    assert rel[0] >= 1
    gen1, set1 = RS.read(a0)
    assert gen1 >= 1 and set1
    # one follower round (READ-ROOM tick -> apply_if_new): the set is applied, the slot frees
    for r in (1, 2):
        RS.apply_if_new(trees[r], pools[r][0])
    assert all(int(x) == 0 for x in a0.slot_refs(set1))
    freed = [h for h in ("a0", "a1", "a2", "a3") if T._refs(a0, p0, [h]) == [0]]
    assert freed
    # 2) the freed page is adopted again on every stage; the next decision never takes it again
    for t, (p, a) in zip(trees, pools):
        T.F._mirror(t, p, a, [[freed[0]]])
    _serve_round(trees, pools, (0, 1, 2), ("r", 2))
    gen2, set2 = RS.read(a0)
    assert gen2 == gen1 + 1 and T._slot(a0, p0, freed[0]) not in set2
    assert T._refs(a0, p0, [freed[0]]) == [3]
    # 3) PP0 has nothing left to give: PP1's private page (c0) is freed by PP1 itself, no shared one
    for n in T._all_nodes(trees[0]):
        n.component_data[FULL].lock_ref = 1
    shared_before = {h: T._refs(a0, p0, [h])[0] for h in ("a0", "a1", "a2", "a3")}
    _serve_round(trees, pools, (0, 1, 2), ("r", 3))
    assert T._refs(a0, p0, ["c0"]) == [0]
    for h, before in shared_before.items():
        assert T._refs(a0, p0, [h])[0] == before, (h, before)


def test_a8_undecided_is_the_verdict_while_pp0_waits_for_an_acknowledgement(pp3, monkeypatch):
    """PP0 published, a follower has not applied yet: PP0's next serve releases nothing and is UNDECIDED
    (a F2 verdict 'did not look', never 'nothing to release'); after the follower's round it decides again."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    monkeypatch.setattr(RR, "deficit", lambda arena, need: (int(need), {
        "slots": 8, "complete": 0, "claimed": 0, "pinned": 0, "free": 0, "clock": 0}))
    _serve_round(trees, pools, (0,), ("u", 1))
    assert _serve_round(trees, pools, (0,), ("u", 2))[0] == 0 and RS.undecided(trees[0])
    assert trees[0]._w3_spill_quiet_until == 0.0
    for r in (1, 2):
        RS.apply_if_new(trees[r], pools[r][0])
    _serve_round(trees, pools, (0,), ("u", 3))
    assert not RS.undecided(trees[0])
