"""Vision on D in the flip layouts (user order 27.10.) -- arming and the scheduler pass of group D.

Hermetic: fake schedulers (one per TP rank), an in-memory wire for the two collectives, the leader's
stage injected. The lockstep is stood in for by running the leader first, then the followers.

Pinned:
  * the pass is armed on group D of a transient boot ONLY outside the dual layout and with the
    switch on (dual: vision stays on P);
  * a pass with candidates attaches rows on every rank and holds nothing; a pass without candidates
    touches no collective;
  * a refused stage holds the request on EVERY rank (same queue mutation), the abort goes to the
    origin from the leader only, and the held request is not planned again;
  * a request whose rows are attached is NOT an image D has to encode -- the W123 guard must admit it
    (regression: the guard counted every placeholder, so a staged image was still refused);
  * the hold is undone by ``vision_unpark`` at the request's position.
"""

import types

import pytest
import torch

from flliper.srt.pdflip import vision_d_guard as vdg
from flliper.srt.pdflip import vision_d_runner as vdr
from flliper.srt.pdflip import vision_rank_runner as vrr

WIDTH = 8


class _Wire:
    def __init__(self):
        self.objs, self.rows = [], []


class _Comm:
    def __init__(self, wire, leader, forbid=False):
        self.wire, self.leader, self.forbid, self._o, self._r = wire, leader, forbid, 0, 0

    def broadcast_obj(self, obj):
        assert not self.forbid, "a collective was entered without a candidate"
        if self.leader:
            self.wire.objs.append(obj)
            return obj
        out = self.wire.objs[self._o]
        self._o += 1
        return out

    def broadcast_rows(self, rows):
        assert not self.forbid
        if self.leader:
            self.wire.rows.append(rows.clone())
            return rows
        rows.copy_(self.wire.rows[self._r])
        self._r += 1
        return rows


def _item(end=9):
    return types.SimpleNamespace(feature=torch.zeros(1), precomputed_embeddings=None, offsets=[(end - 3, end)])


def _req(rid, n=1, streamed=0):
    return types.SimpleNamespace(rid=rid, multimodal_inputs=types.SimpleNamespace(
        mm_items=[_item() for _ in range(n)]), prefix_indices=None, host_hit_length=0,
        output_ids=[7] * streamed, full_untruncated_fill_ids=list(range(120)))


def _sched(queue, leader, aborts=None):
    return types.SimpleNamespace(
        waiting_queue=list(queue), _pdflip_vision_d_leader=leader, _pdflip_vision_refused=set(),
        _pdflip_vision_origin_aborts=[] if aborts is None else aborts, _pdflip_vision_runs=0,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(
            vision_config=types.SimpleNamespace(out_hidden_size=WIDTH))))


def _stage_ok(scheduler, reqs):
    g = torch.Generator().manual_seed(2)
    for r in reqs:
        for it in r.multimodal_inputs.mm_items:
            it.precomputed_embeddings = torch.randn(4, WIDTH, generator=g).to(torch.bfloat16)
    return True, "", ""


def _stage_refuse(scheduler, reqs):
    return False, "W105b PdFlipVisionVictimShort", "the card's air is short"


@pytest.mark.parametrize("env,want", [
    ({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_VISION": "transient"}, True),
    ({"FLLIPER_PDFLIP_GROUP": "P", "FLLIPER_PDFLIP_VISION": "transient"}, False),
    ({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_VISION": "off"}, False),
    ({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_VISION": "transient", "FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}, False),
    ({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_VISION": "transient", "FLLIPER_PDFLIP_VISION_D": "0"}, False),
])
def test_the_d_stage_is_armed_on_group_d_of_a_transient_flip_boot_only(env, want):
    cfg = types.SimpleNamespace(vision_config=object())
    assert vdr.d_stage_wanted(cfg, env) is want
    assert vdr.d_stage_wanted(types.SimpleNamespace(vision_config=None), env) is False


def test_a_pass_attaches_rows_on_every_rank_and_holds_nothing():
    wire = _Wire()
    lead = _sched([_req("a"), _req("b", 2)], True)
    fol = [_sched([_req("a"), _req("b", 2)], False) for _ in range(2)]
    assert vdr.d_pass(lead, comm=_Comm(wire, True), stage=_stage_ok) == []
    for f in fol:
        assert vdr.d_pass(f, comm=_Comm(wire, False)) == []
        got = [it.precomputed_embeddings for r in f.waiting_queue for it in r.multimodal_inputs.mm_items]
        want = [it.precomputed_embeddings for r in lead.waiting_queue for it in r.multimodal_inputs.mm_items]
        assert all(torch.equal(a, b) for a, b in zip(got, want)) and len(got) == 3
        assert [r.rid for r in f.waiting_queue] == ["a", "b"]


def test_a_pass_without_candidates_touches_no_collective():
    s = _sched([types.SimpleNamespace(rid="t", multimodal_inputs=None)], True)
    assert vdr.d_pass(s, comm=_Comm(_Wire(), True, forbid=True)) == []


def test_a_refused_stage_holds_the_request_on_every_rank_and_the_abort_is_the_leaders():
    wire = _Wire()
    lead = _sched([_req("a"), _req("b")], True)
    fol = _sched([_req("a"), _req("b")], False)
    parked_l = vdr.d_pass(lead, comm=_Comm(wire, True), stage=_stage_refuse)
    parked_f = vdr.d_pass(fol, comm=_Comm(wire, False))
    assert [r.rid for _i, r in parked_l] == [r.rid for _i, r in parked_f] == ["a", "b"]
    assert lead.waiting_queue == [] and fol.waiting_queue == []
    assert len(lead._pdflip_vision_origin_aborts) == 2 and fol._pdflip_vision_origin_aborts == []
    assert "W105b" in lead._pdflip_vision_origin_aborts[0][1]
    vrr.vision_unpark(lead, parked_l)
    assert [r.rid for r in lead.waiting_queue] == ["a", "b"]
    # the held rids are not planned again: no collective on the next pass
    again = list(lead.waiting_queue)
    parked2 = vdr.d_pass(lead, comm=_Comm(_Wire(), True, forbid=True))
    assert [(i, r.rid) for i, r in parked2] == [(0, "a"), (1, "b")] and again[0] is parked2[0][1]


def test_the_guard_admits_an_image_whose_rows_are_attached():
    """Regression: ``image_spans`` counted every placeholder, so a D that had just encoded an image
    still refused it by name (W123) at the admission."""
    r = _req("a")
    assert vdg.image_spans(r) != []
    r.multimodal_inputs.mm_items[0].precomputed_embeddings = torch.zeros(4, WIDTH)
    assert vdg.image_spans(r) == [] and vdg.image_end(r) == 0 and vdg.verdict(r, covered=0) == vdg.ADMIT


def test_a_refused_stage_sends_an_unstreamed_request_back_through_p_and_names_a_streamed_one():
    """The abort text is the front's X-REQUEUE key (the W50 marker with the extent sentence) for a
    request that has sent no byte: P has the tower. A request that already streamed keeps the stage's
    named code -- RESUME-VIA-P's P leg carries no image, so a requeue would lose it."""
    wire = _Wire()
    lead = _sched([_req("fresh"), _req("streamed", streamed=3)], True)
    fol = _sched([_req("fresh"), _req("streamed", streamed=3)], False)
    vdr.d_pass(lead, comm=_Comm(wire, True), stage=_stage_refuse)
    vdr.d_pass(fol, comm=_Comm(wire, False))
    msgs = dict(lead._pdflip_vision_origin_aborts)
    assert "PdFlipTpPrefillExceeded" in msgs["fresh"] and "extent after prefix matching is 120" in msgs["fresh"]
    assert "W105b" in msgs["fresh"]
    assert msgs["streamed"].startswith("W105b") and "PdFlipTpPrefillExceeded" not in msgs["streamed"]
