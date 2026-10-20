"""h16t joint probe READ-ROOM x F1 x F2 (F1 review A8, pgap review A1-A7): ONE real C arena, three trees
(pp_size=3), the real ``read_room.serve`` -> ``_w3_arena_spill`` -> ``w3_room_set`` path, and the real F2 verdicts
(``room_verdicts``) of every stage -- followers served FIRST, PP0's set applied by the followers, a page freed and
adopted again, a slot only a follower holds. The question: does F2 ever end a wait while F1 can still free room?

* PP0 decided and waits for a follower's acknowledgement: its serve is UNDECIDED -> verdict 'did not look' -> no
  futility (merge line ``_quiet or _w3rs.undecided(cache)``).
* A follower frees a slot only it holds -> released > 0 -> no futility.
* Futility only when ALL THREE stages answered after the last progress with nothing released, nothing undecided,
  nothing in flight -- two of three never suffice.
The header deficit is pinned (the fixture arena is not full); the three stages share one process here, so each
stage's verdict carries its own stand-in pid."""
import os
import shutil
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_pdflip_f1_w3_room_set_1016 as T  # noqa: E402

from flliper.srt.pdflip import read_room as RR  # noqa: E402
from flliper.srt.pdflip import room_verdicts as RV  # noqa: E402
from flliper.srt.pdflip import room_wait_arbiter as ARB  # noqa: E402
from flliper.srt.pdflip import w3_room_set as RS  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
pp3 = T.pp3
FULL = T.FULL
PIDS = {0: 4100, 1: 4101, 2: 4102}


@pytest.fixture
def joint(pp3, monkeypatch):
    RV.reset_for_tests()
    ARB.reset_for_tests()
    pools, trees, nodes, store = pp3
    monkeypatch.setattr(RR, "deficit", lambda arena, need: (int(need), {
        "slots": 8, "complete": 0, "claimed": 0, "pinned": 0, "free": 0, "clock": 0}))
    monkeypatch.setattr(ARB, "_alive", lambda pid: True)
    cur = {"rank": 0}
    orig = RV.note_verdict

    def per_stage(arena, *a, **k):
        k.setdefault("pid", PIDS[cur["rank"]])
        return orig(arena, *a, **k)

    monkeypatch.setattr(RV, "note_verdict", per_stage)

    def serve(order, key):
        out = {}
        for r in order:
            cur["rank"] = r
            trees[r]._w3_spill_quiet_until = 0.0
            out[r] = RR.serve(trees[r], pools[r][0], pools[r][1], 1, read_key=key)
            time.sleep(0.002)
        return out

    yield pools, trees, serve
    RV.reset_for_tests()
    ARB.reset_for_tests()


def _futile(a0, since):
    return RV.futile(a0, since, pp_size=3)


def test_joint_undecided_and_follower_releases_never_end_a_wait(joint):
    pools, trees, serve = joint
    p0, a0 = pools[0]
    since = ARB.now_ms()
    time.sleep(0.005)
    # followers first, then PP0 decides and releases its share
    rel = serve((1, 2, 0), ("j", 1))
    assert rel[0] >= 1
    assert _futile(a0, since) is None, "PP0 released pages"
    # PP0 asked again before the followers applied its set: UNDECIDED -> no futility although it released 0
    since = ARB.now_ms()
    time.sleep(0.005)
    assert serve((0,), ("j", 3))[0] == 0 and RS.undecided(trees[0])
    for r in (1, 2):                                        # the followers answer 'nothing' (decided, nothing to give)
        RV.note_verdict(pools[r][1], 0, 0, False, pid=PIDS[r])
    assert _futile(a0, since) is None, "an undecided group spill is no answer"
    # the followers apply PP0's set (READ-ROOM tick -> apply_if_new); the slots free on every stage
    for r in (1, 2):
        RS.apply_if_new(trees[r], pools[r][0])
    gen, sset = RS.read(a0)
    assert sset and all(int(x) == 0 for x in a0.slot_refs(sset))


def test_joint_a_follower_only_slot_is_freed_by_it_and_counts_as_released(joint):
    pools, trees, serve = joint
    p0, a0 = pools[0]
    for n in T._all_nodes(trees[0]):                        # PP0 has nothing to give: its set stays empty
        n.component_data[FULL].lock_ref = 1
    since = ARB.now_ms()
    time.sleep(0.005)
    rel = serve((0, 1, 2), ("k", 2))
    assert T._refs(a0, p0, ["c0"]) == [0], "PP1 freed the slot only it held"
    assert rel[1] >= 1
    assert _futile(a0, since) is None, "a follower released pages"


def test_joint_futility_needs_all_three_stages_with_nothing(joint):
    pools, trees, serve = joint
    p0, a0 = pools[0]
    serve((1, 2, 0), ("f", 1))
    for r in (1, 2):
        RS.apply_if_new(trees[r], pools[r][0])
    for t in trees:                                         # nothing is a candidate anywhere any more
        for n in T._all_nodes(t):
            n.component_data[FULL].lock_ref = 1
    serve((0, 1, 2), ("f", 2))                              # drains what is left (PP1's own slot)
    since = ARB.now_ms()
    time.sleep(0.005)
    serve((0, 1), ("f", 3))
    assert _futile(a0, since) is None, "two of three stages never suffice"
    serve((2,), ("f", 4))
    assert not any(RS.undecided(t) for t in trees)
    ev = _futile(a0, since)
    assert ev is not None and all(("pid%d" % p) in ev for p in PIDS.values()), ev


def test_joint_readoption_is_not_released_twice_and_keeps_the_wait(joint):
    """A page freed and adopted again on every stage is not taken again by PP0's next set (F1 review P1); the
    round that frees nothing new because PP0 waits for acknowledgements stays undecided -- no futility."""
    pools, trees, serve = joint
    p0, a0 = pools[0]
    serve((1, 2, 0), ("r", 1))
    for r in (1, 2):
        RS.apply_if_new(trees[r], pools[r][0])
    freed = [h for h in ("a0", "a1", "a2", "a3") if T._refs(a0, p0, [h]) == [0]]
    assert freed
    for t, (p, a) in zip(trees, pools):
        T.F._mirror(t, p, a, [[freed[0]]])
    since = ARB.now_ms()
    time.sleep(0.005)
    serve((0, 1, 2), ("r", 2))
    gen, sset = RS.read(a0)
    assert T._slot(a0, p0, freed[0]) not in sset and T._refs(a0, p0, [freed[0]]) == [3]
    serve((0,), ("r", 3))
    if RS.undecided(trees[0]):
        assert _futile(a0, since) is None
