"""NF-RC31 A (rc30 review, Folgefix A): a STAGED KV-COMPACT plan whose agreed digest equals the last staged
pass's does not repeat its ``st == 1`` evictions -- it is refused by name (``verdict=staged-repeat``).

THE METAL (NF rc30, live 21.10. 07:31:33 / 07:31:38): ``KV-COMPACT ... verdict=staged`` with the same ``digest=``
pass after pass; TP0 ``queued_here=1`` three times, TP1/TP2 ``queued_here=0`` -- the hand-off never left the
devices, and every pass cleared the device again (the agreed L2 evictions) for no stage.

Danger direction "Ränge uneins": the skip must be decided from the replicated digest, never from TP0's queue
count (TP0 queues, a worker queues nothing). Mutant: deciding on ``queued_here`` parts the ranks.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import forma_l3_subtree as S39  # noqa: E402

from test_kvziel_parked_store_1021 import P, _store, _tick_world  # noqa: E402
from test_s77_kv_compact_s9_hold_1019 import FULL, _node  # noqa: E402


def _mixed_world(monkeypatch):
    """One L2-backed idle leaf (``st == 1``: evicted now) and one store-backed leaf (``st == 2``: S39 hand-off),
    both needed as room for the boundary at page 16."""
    alloc, tree, x, sched, ms = _tick_world(monkeypatch)        # x: store-backed [20, 21]
    a = _node(tree, alloc, [22, 23], P)                         # L2-backed (host copy kept)
    assert a.component_data[FULL].host_value is not None
    return alloc, tree, x, a, sched, ms


def _drive(sched, ms, fr, passes=3):
    """Run passes until this one returns from the staged branch (the snapshot lands in a later pass)."""
    before = (int(fr.get("compact_staged", 0)), int(fr.get("compact_staged_repeat", 0)))
    for _ in range(passes):
        dsv._kv_compact_for_shrink(sched, ms, fr, 0, True, 22 * P)
        now = (int(fr.get("compact_staged", 0)), int(fr.get("compact_staged_repeat", 0)))
        if now != before:
            return now
    return before


@pytest.fixture
def evicts(monkeypatch):
    """The agreed evictions, counted; the tree is left as it was -- the content came back (a read loaded it)
    before the next pass, the metal's repeated device clearing."""
    calls = []

    def _ev(cache, plan):
        calls.append([getattr(m.node, "id", None) for m in plan.evicts])
        return len(plan.evicts), 0

    monkeypatch.setattr(KC, "execute_evicts", _ev)
    return calls


def test_a_staged_plan_with_the_same_digest_does_not_repeat_its_evictions(monkeypatch, caplog, evicts):
    caplog.set_level(logging.INFO)
    alloc, tree, x, a, sched, ms = _mixed_world(monkeypatch)
    got = []
    monkeypatch.setattr(S39, "queue_for_room", lambda t, nodes, why="": got.append(list(nodes)) or len(nodes))
    fr = {}
    assert _drive(sched, ms, fr) == (1, 0)
    assert evicts == [[a.id]], evicts                           # pass 1: the L2 leaf goes, x handed over
    memo = fr[dsv.KV_STAGED_MEMO]
    assert memo["handoff_nodes"] == 1 and memo["queued_here"] == 1
    assert _drive(sched, ms, fr) == (1, 1)                      # pass 2: same digest -> staged-repeat
    assert evicts == [[a.id]], evicts                           # no second device clearing
    assert got == [[x], [x]]                                    # the hand-off is re-offered (idempotent queue)
    rep = [m for m in caplog.messages if "verdict=staged-repeat" in m]
    assert rep and "digest=%d" % memo["digest"] in rep[0] and "skipped_evict_nodes=1" in rep[0], rep
    assert "compact_refused" not in fr                          # no refusal memo: the next pass looks again


def test_a_changed_digest_stages_again(monkeypatch, evicts):
    """The hand-off landed (x left the device on every rank): the plan changes, the memo does not hold it back."""
    alloc, tree, x, a, sched, ms = _mixed_world(monkeypatch)
    monkeypatch.setattr(S39, "queue_for_room", lambda t, nodes, why="": len(nodes))
    fr = {}
    assert _drive(sched, ms, fr) == (1, 0)
    d1 = fr[dsv.KV_STAGED_MEMO]["digest"]
    fr[dsv.KV_STAGED_MEMO]["digest"] = d1 ^ 1                    # another staged plan was the last one
    assert _drive(sched, ms, fr) == (2, 0)
    assert len(evicts) == 2


def test_a_non_staged_pass_clears_the_memo(monkeypatch):
    alloc, tree, n, sched, ms = _tick_world(monkeypatch)
    n.component_data[FULL].host_value = n.component_data[FULL].value.clone()   # L2-backed: room now, no stage
    n.l3_present = False
    fr = {dsv.KV_STAGED_MEMO: {"digest": 7, "handoff_nodes": 1, "queued_here": 1, "n": 1, "repeats": 0}}
    for _ in range(3):
        if dsv._kv_compact_for_shrink(sched, ms, fr, 0, True, 22 * P) is not None:
            break
    assert dsv.KV_STAGED_MEMO not in fr


def _ranks(monkeypatch, decide_on_queue=False):
    """Three ranks, one agreed digest: TP0's S39 queue answers 1, a worker's 0 (the live 07:31 shape)."""
    out = []
    for tp0 in (True, False, False):
        S39._reset_for_tests()
        alloc, tree, x, a, sched, ms = _mixed_world(monkeypatch)
        calls = []
        monkeypatch.setattr(KC, "execute_evicts", lambda c, p, calls=calls: calls.append(1) or (len(p.evicts), 0))
        monkeypatch.setattr(S39, "queue_for_room",
                            lambda t, nodes, why="", tp0=tp0: len(nodes) if tp0 else 0)
        fr = {}
        _drive(sched, ms, fr)
        if decide_on_queue and not fr[dsv.KV_STAGED_MEMO]["queued_here"]:
            # MUTANT: "nothing reached S39 here" read from this rank's own queue count
            fr[dsv.KV_STAGED_MEMO]["digest"] = -1
        _drive(sched, ms, fr)
        out.append((len(calls), int(fr.get("compact_staged_repeat", 0))))
    S39._reset_for_tests()
    return out


def test_every_rank_skips_alike_although_only_tp0_queued(monkeypatch):
    assert _ranks(monkeypatch) == [(1, 1)] * 3


def test_mutant_deciding_on_the_queue_count_parts_the_ranks(monkeypatch):
    got = _ranks(monkeypatch, decide_on_queue=True)
    assert got[0] != got[1] and got[1] == got[2], got
