"""Vision on D in the flip layouts (user order 27.10.) -- the pure pass: what the TP ranks of
group D agree on and exchange around the one rank that encodes (TP0, the 5090).

Hermetic: the collectives are an in-memory wire; the leader runs first, then each
follower reads what the leader put on it (sequential stand-in for the lockstep).

Pinned (each case a distinct silent-failure path):
  * an empty plan makes NO collective -- a rank entering a broadcast the others skip is a hang;
  * followers receive the leader's rows bitwise and attach them to THEIR items, width-checked;
  * a refused leader moves no rows and attaches nothing anywhere;
  * a follower whose queue differs from the leader's attaches NOTHING (positional attach on a
    different queue = one image's rows on another image's placeholders, fluent wrong text);
  * a stage that says ok but attached no rows is a named verdict, not an empty broadcast.
"""

import types

import pytest
import torch

from sglang.srt.planner.vision_stage_load import VisionEmbeddingRefused, attach_precomputed_embeddings
from sglang.srt.weg2 import vision_d_pass as vdp

WIDTH = 8


class _Wire:
    def __init__(self):
        self.objs, self.rows, self.calls = [], [], 0


class _Comm:
    def __init__(self, wire, leader):
        self.wire, self.leader, self._o, self._r = wire, leader, 0, 0

    def broadcast_obj(self, obj):
        self.wire.calls += 1
        if self.leader:
            self.wire.objs.append(obj)
            return obj
        out = self.wire.objs[self._o]
        self._o += 1
        return out

    def broadcast_rows(self, rows):
        self.wire.calls += 1
        if self.leader:
            self.wire.rows.append(rows.clone())
            return rows
        src = self.wire.rows[self._r]
        self._r += 1
        rows.copy_(src)
        return rows


def _item():
    return types.SimpleNamespace(feature=torch.zeros(1), precomputed_embeddings=None)


def _req(rid, n_items):
    return types.SimpleNamespace(rid=rid, items=[_item() for _ in range(n_items)])


def _unstaged(req):
    return [it for it in req.items if it.precomputed_embeddings is None]


def _plan(queue, refused=()):
    return vdp.plan(queue, unstaged_items=_unstaged, refused=refused)


def _leader_run(queue, wire, ok=True, code="", attach_rows=True):
    entries = _plan(queue)
    g = torch.Generator().manual_seed(5)
    for k, (_r, _i, it) in enumerate(entries):
        if ok and attach_rows:
            it.precomputed_embeddings = torch.randn(3 + k, WIDTH, generator=g).to(torch.bfloat16)
    packet, rows = vdp.leader_packet(entries, ok=ok, code=code, detail="boom" if not ok else "")
    got, mine = vdp.exchange(entries, packet, rows, is_leader=True, comm=_Comm(wire, True), width=WIDTH)
    return entries, got, mine


def _follower_run(queue, wire):
    entries = _plan(queue)
    got, rows = vdp.exchange(entries, None, [], is_leader=False, comm=_Comm(wire, False), width=WIDTH)
    n = vdp.attach_followers(entries, rows, attach=attach_precomputed_embeddings, width=WIDTH) if got.ok else 0
    return entries, got, rows, n


def test_an_empty_plan_makes_no_collective():
    wire = _Wire()
    assert _plan([_req("a", 0)]) == []
    # the orchestration contract: no entries -> the caller returns before exchange; attach of nothing is 0
    assert vdp.attach_followers([], [], attach=attach_precomputed_embeddings, width=WIDTH) == 0
    assert wire.calls == 0


def test_followers_get_the_leaders_rows_bitwise_and_attach_them_to_their_own_items():
    wire = _Wire()
    lq = [_req("a", 2), _req("b", 1)]
    fq = [_req("a", 2), _req("b", 1)]            # another rank, same queue, its own item objects
    _entries, got, rows = _leader_run(lq, wire)
    assert got.ok and [(w.rid, w.idx, w.rows) for w in got.wanted] == [("a", 0, 3), ("a", 1, 4), ("b", 0, 5)]
    entries, got_f, rows_f, n = _follower_run(fq, wire)
    assert n == 3 + 4 + 5
    for (_r, _i, it), lead_rows in zip(entries, rows):
        assert torch.equal(it.precomputed_embeddings, lead_rows)


def test_a_refused_leader_moves_no_rows_and_attaches_nothing():
    wire = _Wire()
    lq, fq = [_req("a", 1)], [_req("a", 1)]
    _e, got, rows = _leader_run(lq, wire, ok=False, code="W105b")
    assert not got.ok and got.code == "W105b" and rows == [] and wire.rows == []
    entries, got_f, rows_f, n = _follower_run(fq, wire)
    assert not got_f.ok and n == 0 and entries[0][2].precomputed_embeddings is None


def test_a_follower_with_another_queue_attaches_nothing():
    wire = _Wire()
    _leader_run([_req("a", 1), _req("b", 1)], wire)
    fq = [_req("a", 1)]                          # this rank never saw request b
    fe = _plan(fq)
    with pytest.raises(vdp.DisagreeError, match="W124"):
        vdp.exchange(fe, None, [], is_leader=False, comm=_Comm(wire, False), width=WIDTH)
    assert fe[0][2].precomputed_embeddings is None


def test_a_stage_that_attached_no_rows_is_a_named_verdict():
    wire = _Wire()
    _e, got, rows = _leader_run([_req("a", 1)], wire, ok=True, attach_rows=False)
    assert not got.ok and got.code == vdp.W_NO_ROWS and rows == []


def test_a_width_mismatch_stops_the_follower_by_name():
    wire = _Wire()
    _leader_run([_req("a", 1)], wire)
    fq = [_req("a", 1)]
    fe = _plan(fq)
    got, rows = vdp.exchange(fe, None, [], is_leader=False, comm=_Comm(wire, False), width=WIDTH)
    with pytest.raises(VisionEmbeddingRefused, match="width"):
        vdp.attach_followers(fe, rows, attach=attach_precomputed_embeddings, width=WIDTH + 1)
