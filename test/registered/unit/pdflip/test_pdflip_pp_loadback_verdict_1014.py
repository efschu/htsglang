"""LB-VERDICT-1014 (NF h16 a2cfc77f76, P, 14.10. 17:30:29Z, P.log ..._1014_164206 158839-158959).

``#1004 SLOT DISAGREEMENT: PP2 is launching slot 1 (fwd_ct=87, rids=[pdflip-40-310], extend=275) but the upstream's
proxy names slot 2``. pdflip-40-310's host load-back (kv_tokens=150720) was decided on every stage's OWN pool:
PP2 dropped an un-backed leaf (EVICT-UNBACKED-DROP freed=118720) and loaded in pass N; PP0/PP1 could not (PP0
UD-REFUSED child_host_locked; 'SF LOADBACK-ROOM PP-RESIDUAL avail_after=140416') and loaded in pass N+1.

The rule pinned here: PP0 decides, the request chain carries the verdict, every stage loads in the pass that
absorbs it; a stage that cannot pay a released load-back stops by name. The RANK-EQUALITY MUTANT is the old
rank-local verdict (switch off): it reproduces the h16 split.
"""

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.pdflip import pp_loadback_verdict as LBV
from flliper.srt.pdflip.pp_slot_fidelity import FLOOR_LOCAL_PP_ATTR

RID = "pdflip-40-310"
KV = 150720


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.delenv(LBV.ENV, raising=False)


def _stage(rank):
    tree = types.SimpleNamespace()
    setattr(tree, FLOOR_LOCAL_PP_ATTR, True)
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=rank), tree_cache=tree)
    assert LBV.arm(sched)
    return tree


def _pass(trees, room, intake, gate=True):
    """One pass on three stages. ``room[k]`` = stage k's own room verdict this pass; ``intake`` = the verdicts PP0
    put on the wire at this pass's intake (dispatched on every stage BEFORE planning). Returns who loaded."""
    for t in trees:
        for v in intake:
            LBV.absorb(t, v)
    loaded = []
    for k, t in enumerate(trees):
        if not gate:                      # the h16 code: each stage's own room decides
            if room[k]:
                loaded.append(k)
            continue
        if not LBV.released(t, RID):
            LBV.hold(t, RID, KV, lambda k=k: room[k])
            continue
        if not room[k]:
            raise RuntimeError(f"{LBV.MARK} SHORT stage={k}")  # the named stop of load_back
        LBV.consume(t, RID)
        loaded.append(k)
    return loaded, LBV.take(trees[0])


def test_h16_one_decision_one_pass_every_stage():
    trees = [_stage(0), _stage(1), _stage(2)]
    # pass N: PP2 has room (its UD drop), PP0/PP1 do not -- nobody loads, PP0 sends nothing
    loaded, out = _pass(trees, [False, False, True], [])
    assert loaded == [] and out == []
    # pass N+1: PP0's room is there now -> its verdict goes on the wire; still nobody loads
    loaded, out = _pass(trees, [True, True, True], [])
    assert loaded == [] and [v.rid for v in out] == [RID]
    # pass N+2: every stage absorbs it at the intake and loads in THIS pass
    loaded, out = _pass(trees, [True, True, True], out)
    assert loaded == [0, 1, 2] and out == []
    assert not any(LBV.released(t, RID) for t in trees)


def test_rank_equality_mutant_the_rank_local_verdict_splits_the_stages():
    """The h16 code (each stage's own room): PP2 loads in pass N, PP0/PP1 in N+1 -> #1004."""
    trees = [_stage(0), _stage(1), _stage(2)]
    n, _ = _pass(trees, [False, False, True], [], gate=False)
    n1, _ = _pass(trees, [True, True, False], [], gate=False)
    assert n == [2] and n1 == [0, 1], "the detector must see the h16 split"


def test_a_follower_never_decides_and_never_sends():
    trees = [_stage(0), _stage(1), _stage(2)]
    LBV.hold(trees[2], RID, KV, lambda: True)
    assert LBV.take(trees[2]) == [] and not LBV.released(trees[2], RID)


def test_a_released_load_back_a_stage_cannot_pay_stops_by_name():
    trees = [_stage(0), _stage(1), _stage(2)]
    _pass(trees, [True, True, True], [])
    out = [LBV.PdFlipPpLoadbackVerdict(rid=RID)]
    with pytest.raises(RuntimeError, match="LB-VERDICT SHORT stage=1"):
        _pass(trees, [True, False, True], out)


def test_pp0_sends_a_verdict_once():
    t = _stage(0)
    LBV.hold(t, RID, KV, lambda: True)
    LBV.hold(t, RID, KV, lambda: True)
    assert [v.rid for v in LBV.take(t)] == [RID]
    assert LBV.take(t) == []


def test_not_armed_off_the_local_pp_floor_or_switch_off(monkeypatch):
    t = _stage(0)
    setattr(t, FLOOR_LOCAL_PP_ATTR, False)
    assert not LBV.armed(t)
    setattr(t, FLOOR_LOCAL_PP_ATTR, True)
    assert LBV.armed(t)
    monkeypatch.setenv(LBV.ENV, "0")
    assert not LBV.armed(t)
    one = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=1, pp_rank=0), tree_cache=types.SimpleNamespace())
    assert not LBV.arm(one)


def test_load_back_holds_before_any_room_verdict_and_stops_on_a_released_short():
    """The production seam: in UnifiedRadixCache.load_back the gate sits after the agreed floor and BEFORE
    local_pp_room / the floor refusal; a released load-back whose room fails raises AgreedRoomShort."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.load_back)
    i_gate = src.index("_lbv.hold(self, _lbv_rid")
    i_room = src.index("_room = _sf.local_pp_room(self, kv_tokens, floor")
    i_floor = src.index("if floor < kv_tokens:")
    assert i_gate < i_room < i_floor
    assert "elif _lbv_gated:" in src and "raise _sf.AgreedRoomShort(" in src
    assert "_lbv.consume(self, _lbv_rid)" in src


def test_the_scheduler_dispatches_and_relays_the_verdict():
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S)
    assert "(PdFlipPpLoadbackVerdict, self.handle_pdflip_pp_loadback_verdict)" in src
    assert "_lbv_compose(self, _vision_origin_aborts)" in src


def test_default_on():
    assert LBV.enabled()
