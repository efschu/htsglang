"""PF late retract (item 220, residual of item 180 5220e4fe47).

After the Admit a follower whose live tree stays short of told is held
(item 180) and re-asks PP0 on the PF ack stream -- but PP0 had dropped the
rid's open state at the Admit, so the ask died as "PF TOLD-ACK LATE" and the
hold had no live answer. Now PP0 retracts the rid to told=0 for EVERY rank on
the Admit channel (fallback marker) while it has not seated the rid itself:

  * only PP0 decides, from PP0-local facts (late ack != admitted told, rid
    still in PP0's own waiting queue); no rank-local admit, no reserve;
  * PP0 seated the rid already -> nothing on the wire (TOO LATE / LATE), a
    retract then would itself be the split;
  * PP0 never waits for a follower, so the hold cannot deadlock against it.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402,F401

import _told_ring_pf as R  # noqa: E402,F401
from sglang.srt.managers import weg2_store_told as m  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.weg2 import p_intake  # noqa: E402
from sglang.srt.weg2 import p_twin_defer as twin  # noqa: E402

from test_weg2_pp_width_uniform_1003 import (  # noqa: E402
    HEAD, RID, TOLD, _StageTree, _req, _stage,
)
from test_weg2_told_group_fallback_pf import (  # noqa: E402
    SLOW, _clean_ledgers, _fb_ring, _kinds, _reads,
)

HOLD_UNTIL = 60     # PP0 pass before which no rank seats the rid (kept verdict gap)


def _ring(monkeypatch, hold_until=None):
    ring = _fb_ring(monkeypatch, _reads())
    if hold_until is not None:
        ring.hold = lambda rank, rid, plan: rid == SLOW and plan < hold_until
    ring.arrive(SLOW)
    return ring


def _until_admit_told(ring, limit=200):
    for _ in range(limit):
        ring.step()
        if any(type(o).__name__ == "Weg2StoreAdmit" for o in ring.wire_objs()):
            return
    raise AssertionError("no Admit")


def _late_ask(ring, rank=2, reach=0):
    ring.box.append(fb.Weg2ToldReadAck(rank=rank, seq=99, reads=[(SLOW, reach)]))


def test_late_short_reach_before_pp0_seats_is_retracted_for_every_rank(monkeypatch, caplog):
    ring = _ring(monkeypatch, hold_until=HOLD_UNTIL)
    _until_admit_told(ring)
    assert _kinds(ring)[-1][:2] == ("Weg2StoreAdmit", 4096)
    _late_ask(ring)
    with caplog.at_level("WARNING", logger=fb.logger.name):
        ring.run(160)
    kinds = _kinds(ring)
    assert kinds[-2][:2] == ("Weg2StoreAdmit", 4096)
    assert kinds[-1] == ("Weg2StoreAdmit", 0, None, None, 1), kinds
    plans = ring.plans(SLOW)
    assert all(len(p) == 1 for p in plans), plans
    assert plans[0] == plans[1] == plans[2], "one PP0 pass, one cap, on every rank"
    assert plans[0][0][2] == 0, "the prefix cap is 0 everywhere"
    assert [a[3] for s in ring.stages for a in s.admitted] == [0, 0, 0]
    assert plans[0][0][0] >= HOLD_UNTIL
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}, "nobody waited"
    for s in ring.stages:
        assert SLOW in s.tree_cache.released
    _clean_ledgers(ring, SLOW)
    assert ring.stages[0]._pf_retract_n == 1
    assert len([x for x in caplog.messages if "PF TOLD-RETRACT rid" in x]) == 1
    assert not fb._admitted_map(ring.stages[0]) and not fb._late_map(ring.stages[0])


def test_the_retract_is_one_wire_object_not_a_rank_local_admit(monkeypatch):
    ring = _ring(monkeypatch, hold_until=HOLD_UNTIL)
    _until_admit_told(ring)
    _late_ask(ring, rank=1, reach=0)
    _late_ask(ring, rank=2, reach=17)       # two followers ask in the same window
    ring.run(160)
    retracts = [k for k in _kinds(ring) if k[0] == "Weg2StoreAdmit" and k[4] == 1]
    assert len(retracts) == 1, "PP0 answers once for all ranks"
    plans = ring.plans(SLOW)
    assert plans[0] == plans[1] == plans[2]


def test_late_ask_after_pp0_seated_puts_nothing_on_the_wire(monkeypatch, caplog):
    ring = _ring(monkeypatch)               # PP0 seats in the Admit pass
    _until_admit_told(ring)
    ring.run(3)
    assert ring.stages[0].admitted, "PP0 seated at told"
    n_wire = len(ring.wire_objs())
    _late_ask(ring)
    with caplog.at_level("WARNING", logger=fb.logger.name):
        ring.run(20)
    assert len(ring.wire_objs()) == n_wire, "a retract after PP0's seat is the split itself"
    assert not [k for k in _kinds(ring) if k[4] == 1]
    plans = ring.plans(SLOW)
    assert plans[0] == plans[1] == plans[2] and plans[0][0][2] != 0
    # PP0 stopped watching once it seated the rid: the ask stays unread on the
    # stream (named as ACK LATE whenever the stream is next harvested)
    assert not fb.pp0_watching(ring.stages[0])


def test_too_late_is_named_when_the_ask_lands_the_pass_after_the_seat(monkeypatch, caplog):
    ring = _ring(monkeypatch, hold_until=HOLD_UNTIL)
    _until_admit_told(ring)
    sc = ring.stages[0]
    _late_ask(ring)
    fb.pp0_harvest(sc)                      # the ask is in, PP0 has not published yet
    sc.waiting_queue.clear()                # ...and PP0 seated the rid meanwhile
    with caplog.at_level("ERROR", logger=fb.logger.name):
        assert fb.pp0_retract_due(sc, set(), set()) == []
    assert any("PF TOLD-RETRACT TOO LATE" in x for x in caplog.messages)
    assert not fb._admitted_map(sc) and not fb._late_map(sc)


def test_late_ack_that_reproduces_told_retracts_nothing(monkeypatch):
    ring = _ring(monkeypatch, hold_until=HOLD_UNTIL)
    _until_admit_told(ring)
    _late_ask(ring, reach=4096)
    ring.run(160)
    assert not [k for k in _kinds(ring) if k[4] == 1]
    assert ring.plans(SLOW)[0][0][2] != 0


def test_a_parked_rid_keeps_its_retract_candidacy(monkeypatch):
    ring = _ring(monkeypatch, hold_until=HOLD_UNTIL)
    _until_admit_told(ring)
    sc = ring.stages[0]
    _late_ask(ring)
    fb.pp0_harvest(sc)
    assert fb.pp0_retract_due(sc, set(), {SLOW}) == []
    assert SLOW in fb._late_map(sc) and SLOW in fb._admitted_map(sc)
    assert fb.pp0_retract_due(sc, {SLOW}, set()) == [(SLOW, 4096, 0)]


def test_admitted_table_is_pruned_when_the_rid_leaves(monkeypatch):
    ring = _ring(monkeypatch)
    _until_admit_told(ring)
    ring.run(5)
    assert not fb.pp0_watching(ring.stages[0]), "seated: no harvest kept alive for it"


def test_switch_off_builds_no_retract_state(monkeypatch):
    ring = _fb_ring(monkeypatch, _reads(), on=False)
    ring.arrive(SLOW)
    ring.run(80)
    assert not hasattr(ring.stages[0], "_weg2_fb_admitted")
    assert not hasattr(ring.stages[0], "_weg2_fb_late")


# -- follower half: the hold ends on the retract --------------------------------

def test_held_follower_re_asks_then_admits_at_zero_on_the_retract():
    tree = _StageTree(dev=HEAD, host=TOLD - HEAD, anchor=TOLD, record=TOLD)
    stage = _stage(1, tree, store_has=HEAD)
    stage._weg2_fb_follower = fb._FState()
    stage._weg2_fb_channel = R.MemChannel([])
    req = _req(head=0)
    stage._weg2_store_told[RID] = TOLD
    twin.note_follower_twin(stage, RID)
    note = lambda *a: None  # noqa: E731
    assert p_intake.told_admission(stage, req, note, m.admission) == TOLD
    tree.host, tree.anchor, tree.record = 0, None, 0           # the tree falls short
    stage._w27u_wait_spent_s = {RID: m.WAIT_CAP_S}
    assert p_intake.told_admission(stage, req, note, m.admission) is None
    assert stage._weg2_fb_follower.outbox == [(RID, HEAD)]
    assert m.follower_absorb(stage, []) == []        # the pass pumps the re-ask to PP0
    assert [ack.reads for ack in stage._weg2_fb_channel.box] == [[(RID, HEAD)]]
    retract = m.Weg2StoreAdmit(rid=RID, told=0)
    setattr(retract, fb.WIRE_FALLBACK, 1)
    assert m.follower_absorb(stage, [retract]) == []
    assert stage._weg2_store_told.get(RID) == 0
    assert RID not in (stage._w27u_unreached or {})
    credit = p_intake.told_admission(stage, req, note, m.admission)
    assert credit == 0, "admitted at told=0, like PP0 and every rank"
