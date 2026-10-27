"""Vision on D in the flip layouts (user order 27.10.) -- the pure pass: what the TP ranks of
group D agree on and exchange around the one rank that encodes (TP0, the 5090).

Hermetic: the collectives are an in-memory wire; the leader runs first, then each
follower reads what the leader put on it (sequential stand-in for the lockstep).

Pinned (each case a distinct silent-failure path):
  * an empty plan has no candidates -- the caller returns before any collective (a rank entering a
    broadcast the others skip is a hang);
  * the leader's skip/wait decision (its prefix match) reaches the followers; a follower stages exactly
    what the leader staged;
  * followers receive the rows bitwise and attach them to THEIR items, width-checked;
  * a refused leader moves no rows and attaches nothing anywhere;
  * a follower whose queue lacks something the leader selected does NOTHING (positional attach on a
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
        self.objs, self.rows = [], []


class _Comm:
    def __init__(self, wire, leader):
        self.wire, self.leader, self._o, self._r = wire, leader, 0, 0

    def broadcast_obj(self, obj):
        if self.leader:
            self.wire.objs.append(obj)
            return obj
        out = self.wire.objs[self._o]
        self._o += 1
        return out

    def broadcast_rows(self, rows):
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


def _leader(queue, wire, skip=(), wait=(), ok=True, code="", attach_rows=True):
    entries = _plan(queue)
    sel = vdp.selection_of(entries, skip=skip, wait=wait)
    comm = _Comm(wire, True)
    comm.broadcast_obj(sel)
    chosen = vdp.resolve(entries, sel)
    g = torch.Generator().manual_seed(5)
    for k, (_r, _i, it) in enumerate(chosen):
        if ok and attach_rows:
            it.precomputed_embeddings = torch.randn(3 + k, WIDTH, generator=g).to(torch.bfloat16)
    packet, rows = vdp.leader_packet(chosen, ok=ok, code=code, detail="boom" if not ok else "")
    got, mine = vdp.exchange(chosen, packet, rows, is_leader=True, comm=comm, width=WIDTH)
    return chosen, got, mine, sel


def _follower(queue, wire):
    entries = _plan(queue)
    comm = _Comm(wire, False)
    sel = comm.broadcast_obj(None)
    chosen = vdp.resolve(entries, sel)
    got, rows = vdp.exchange(chosen, None, [], is_leader=False, comm=comm, width=WIDTH)
    n = vdp.attach_followers(chosen, rows, attach=attach_precomputed_embeddings, width=WIDTH) if got.ok else 0
    return chosen, got, rows, n, sel


def test_a_queue_without_unstaged_items_has_no_candidates():
    assert _plan([_req("a", 0)]) == []
    assert vdp.attach_followers([], [], attach=attach_precomputed_embeddings, width=WIDTH) == 0


def test_followers_get_the_leaders_rows_bitwise_and_attach_them_to_their_own_items():
    wire = _Wire()
    lq = [_req("a", 2), _req("b", 1)]
    fq = [_req("a", 2), _req("b", 1)]            # another rank, same queue, its own item objects
    _c, got, rows, _sel = _leader(lq, wire)
    assert got.ok and [(w.rid, w.idx, w.rows) for w in got.wanted] == [("a", 0, 3), ("a", 1, 4), ("b", 0, 5)]
    chosen, _g, _r, n, _s = _follower(fq, wire)
    assert n == 3 + 4 + 5
    for (_r_, _i, it), lead_rows in zip(chosen, rows):
        assert torch.equal(it.precomputed_embeddings, lead_rows)


def test_the_leaders_skip_and_wait_decision_is_what_every_rank_stages():
    """The prefix match that prices a skip is local to the leader: a follower must not stage a
    request the leader skipped (or hold one it staged)."""
    wire = _Wire()
    lq = [_req("a", 1), _req("b", 1), _req("c", 1)]
    fq = [_req("a", 1), _req("b", 1), _req("c", 1)]
    _c, got, _rows, sel = _leader(lq, wire, skip=("a",), wait=("c",))
    assert [w.rid for w in sel.stage] == ["b"] and [w.rid for w in got.wanted] == ["b"]
    chosen, _g, _r, n, fsel = _follower(fq, wire)
    assert [r.rid for r, _i, _it in chosen] == ["b"] and fsel.skip == ("a",) and fsel.wait == ("c",)
    assert fq[0].items[0].precomputed_embeddings is None and fq[2].items[0].precomputed_embeddings is None
    assert fq[1].items[0].precomputed_embeddings is not None and n == 3


def test_a_refused_leader_moves_no_rows_and_attaches_nothing():
    wire = _Wire()
    lq, fq = [_req("a", 1)], [_req("a", 1)]
    _c, got, rows, _sel = _leader(lq, wire, ok=False, code="W105b")
    assert not got.ok and got.code == "W105b" and rows == [] and wire.rows == []
    chosen, got_f, rows_f, n, _s = _follower(fq, wire)
    assert not got_f.ok and n == 0 and chosen[0][2].precomputed_embeddings is None


def test_a_follower_missing_what_the_leader_selected_does_nothing():
    wire = _Wire()
    _leader([_req("a", 1), _req("b", 1)], wire)
    fq = [_req("a", 1)]                          # this rank never saw request b
    comm = _Comm(wire, False)
    sel = comm.broadcast_obj(None)
    with pytest.raises(vdp.DisagreeError, match="W124"):
        vdp.resolve(_plan(fq), sel)
    assert fq[0].items[0].precomputed_embeddings is None


def test_a_follower_that_staged_other_items_than_the_packet_attaches_nothing():
    wire = _Wire()
    _leader([_req("a", 1), _req("b", 1)], wire)
    fq = [_req("a", 1), _req("b", 1)]
    entries = _plan(fq)
    with pytest.raises(vdp.DisagreeError, match="W124"):
        vdp.exchange(entries[:1], None, [], is_leader=False, comm=_Comm(_second_wire(wire), False), width=WIDTH)
    assert all(it.precomputed_embeddings is None for _r, _i, it in entries)


def _second_wire(wire):
    w = _Wire()
    w.objs = wire.objs[1:]        # skip the selection; the packet is next
    w.rows = list(wire.rows)
    return w


def test_a_stage_that_attached_no_rows_is_a_named_verdict():
    wire = _Wire()
    _c, got, rows, _sel = _leader([_req("a", 1)], wire, ok=True, attach_rows=False)
    assert not got.ok and got.code == vdp.W_NO_ROWS and rows == []


def test_a_width_mismatch_stops_the_follower_by_name():
    wire = _Wire()
    _leader([_req("a", 1)], wire)
    fq = [_req("a", 1)]
    comm = _Comm(wire, False)
    sel = comm.broadcast_obj(None)
    chosen = vdp.resolve(_plan(fq), sel)
    _got, rows = vdp.exchange(chosen, None, [], is_leader=False, comm=comm, width=WIDTH)
    with pytest.raises(VisionEmbeddingRefused, match="width"):
        vdp.attach_followers(chosen, rows, attach=attach_precomputed_embeddings, width=WIDTH + 1)
