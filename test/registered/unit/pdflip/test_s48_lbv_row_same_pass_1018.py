# SPDX-License-Identifier: Apache-2.0
"""S48 LBV-ROW-SAME-PASS (NF rc13 ``..._6767e25d89_1018_200305``): on the row form PP0 loads its own fitting host
load-back in the pass that decides it.

THE CASE (rc13 P.log, PP0): pdflip-14-182 ``LB-VERDICT VERDICT`` 20:26:24 -> ``PDFLIP INTAKE-STALL-DEFERRED`` ->
``#1466 PASS-STALL commit_ms=1761`` (the predecessor's last-chunk ring, nothing launched) -> 20:26:26 ``RELEASE`` ->
``#969N ADMIT``; pdflip-19-222 the same with ``commit_ms=2644``. On this boot the followers execute PP0's row
(``P-ROW-AUTHORITY armed``), every follower load-back is row-executed (``LB-VERDICT ROW-ROOM`` n=1..64, 0 ``HOLD``
lines on PP1/PP2): the only stage the verdict pass held was PP0 itself.

THE RULE: row form -> PP0 decides and loads in the same pass; the verdict still rides the next intake (followers
byte-identical), PP0's own absorb releases nothing. Carrierless form, followers, switch off: the hold of before.

Every test drives the real ``UnifiedRadixCache.load_back`` / ``init_load_back`` on the stand-in stages of
``test_pdflip_nf_968_rowlb_1014`` (the metal harness of NF-968-ROWLB), and the real ``pp_loadback_verdict``.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.base_prefix_cache import InitLoadBackParams  # noqa: E402
from flliper.srt.pdflip import pp_loadback_verdict as LBV  # noqa: E402
from flliper.srt.pdflip import pp_slot_fidelity as SF  # noqa: E402

import test_pdflip_nf_968_rowlb_1014 as H  # noqa: E402  (the stand-in stages; PYTHONPATH carries the test dir)

R1, R2 = H.R1, H.R2


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for env in (LBV.ENV, LBV.ROW_YIELD_ENV, H.ROW_ROOM_ENV, SF.ENV, getattr(LBV, "SAME_PASS_ENV", "x_unset")):
        monkeypatch.delenv(env, raising=False)


def _stage(rank, avail, evictable=0, row_form=True):
    s = H._Stage(rank, avail, evictable)
    LBV.state(s).row_form = bool(row_form)       # the arm read the form; the stand-in has no row authority
    return s


def _own_plan(stage, rid):
    _m, _i, best = stage.insert_read(rid)
    return H._Req(rid, best, stage.root_node), best


def _intake(stage):
    """PP0's next intake: take its verdicts and absorb them on every listed stage (PP0 dispatches its own wire too)."""
    out = LBV.take(stage)
    for v in out:
        LBV.absorb(stage, v)
    return out


def test_row_form_pp0_loads_in_the_deciding_pass():
    pp0 = _stage(0, 200000)
    req, best = _own_plan(pp0, R2)
    assert pp0.load_back(best, None, req=req) is True                 # loaded NOW (before S48: False, held)
    assert pp0.cache_controller.loads == [H.LEN[R2]]
    assert getattr(req, "_h105e_floor_refused", False) is False
    # the wire is unchanged: the verdict still rides the next intake
    wire = _intake(pp0)
    assert [v.rid for v in wire] == [R2]
    # PP0's own absorb leaves no release behind (a later load-back of the rid decides afresh)
    assert not LBV.released(pp0, R2) and R2 not in LBV.state(pp0).self_spent and R2 not in LBV.state(pp0).sent
    assert pp0.locks == 1                                             # the ongoing load-back's lock only


def test_row_form_refusal_waits_and_names_its_terms(caplog):
    """PP0's room short even after evicting everything: the request waits (False), nothing on the wire, the locks
    unwound, and the PP0-WAIT line carries kv / floor / agreed R_m / avail / evictable."""
    pp0 = _stage(0, 50000, evictable=10000)
    req, best = _own_plan(pp0, R2)
    with caplog.at_level(logging.INFO, logger=LBV.__name__):
        assert pp0.load_back(best, None, req=req) is False
    assert req._h105e_floor_refused is True
    assert LBV.take(pp0) == []
    assert pp0.locks == 0 and pp0.cache_controller.loads == []
    line = [r.getMessage() for r in caplog.records if "PP0-WAIT" in r.getMessage()]
    assert line and f"rid={R2} kv_tokens={H.LEN[R2]}" in line[0] and "agreed=-" in line[0] and "row_form=1" in line[0]


def test_row_form_shortfall_is_evicted_then_loaded_in_the_same_pass():
    pp0 = _stage(0, 60000, evictable=80000)
    pp0.uniform_avail_floor = 60000
    req, best = _own_plan(pp0, R2)
    assert pp0.load_back(best, None, req=req) is True
    assert pp0.evicted_log == [H.LEN[R2] - 60000]                     # the shortfall only, never the whole set
    assert pp0.cache_controller.loads == [H.LEN[R2]]


def test_agreed_room_still_binds_pp0_on_the_row_form():
    """R_m (PR cap) below the load-back: PP0 refuses this pass exactly as before (every stage reads the same R_m)."""
    pp0 = _stage(0, 200000)
    SF_set = __import__("flliper.srt.pdflip.pp_room_vote", fromlist=["set_cap"]).set_cap
    SF_set(pp0, 1000)
    req, best = _own_plan(pp0, R2)
    assert pp0.load_back(best, None, req=req) is False
    assert LBV.take(pp0) == []


def test_carrierless_form_keeps_the_one_pass_verdict():
    """Followers plan for themselves there: PP0 holds, the verdict is absorbed next intake, then it loads."""
    pp0 = _stage(0, 200000, row_form=False)
    req, best = _own_plan(pp0, R2)
    assert pp0.load_back(best, None, req=req) is False                # held
    assert [v.rid for v in _intake(pp0)] == [R2]
    assert LBV.released(pp0, R2)
    new, _last = pp0.init_load_back(InitLoadBackParams(best_match_node=best, host_hit_length=H.LEN[R2], req=req))
    assert len(new) == H.LEN[R2] and not LBV.released(pp0, R2)


def test_switch_off_is_the_hold_of_before(monkeypatch):
    monkeypatch.setenv(LBV.SAME_PASS_ENV, "0")
    pp0 = _stage(0, 200000)
    req, best = _own_plan(pp0, R2)
    assert pp0.load_back(best, None, req=req) is False
    assert [v.rid for v in _intake(pp0)] == [R2] and LBV.released(pp0, R2)


def test_a_follower_never_decides_alone_on_the_row_form():
    """R11: a follower's OWN-plan load-back (not a row) stays gated and held on every form -- it never loads on its
    own room; only PP0's verdict releases it."""
    fol = _stage(1, 262144)
    assert LBV.same_pass_applies(fol) is False
    req, best = _own_plan(fol, R2)
    assert fol.load_back(best, None, req=req) is False
    assert fol.cache_controller.loads == [] and LBV.take(fol) == []
    LBV.absorb(fol, LBV.PdFlipPpLoadbackVerdict(rid=R2))
    assert LBV.released(fol, R2)                                      # a follower's absorb is unchanged


def test_a_later_load_back_of_the_rid_decides_afresh():
    """RING-ORDER pin: PP0's absorb of a same-pass verdict must not leave a release. A later load-back of the same rid
    (next pass, room gone) then waits on PP0's fresh decision -- never runs on the verdict of an earlier pass, whose
    room change would stop the group by name (LB-VERDICT SHORT). Mutant 'absorb adds the release' -> raises here."""
    pp0 = _stage(0, 200000)
    req, best = _own_plan(pp0, R2)
    assert pp0.load_back(best, None, req=req) is True
    _intake(pp0)
    n = best                                   # the prefix was evicted to the host again (a later turn)
    while n is not None and n.parent is not None:
        n.component_data[H.BASE].value = None
        n = n.parent
    pp0.token_to_kv_pool_allocator.avail = 1000
    pp0.uniform_avail_floor = 1000
    req2 = H._Req(R2, best, pp0.root_node)
    assert pp0.load_back(best, None, req=req2) is False               # waits; no AgreedRoomShort


def test_row_form_read_at_arm_is_safe_on_unreadable_forms():
    from types import SimpleNamespace

    tree = SimpleNamespace()
    assert LBV.arm(SimpleNamespace(ps=SimpleNamespace(pp_size=3, pp_rank=0), tree_cache=tree))
    assert LBV.state(tree).row_form is False                          # no row authority readable -> the hold


def test_row_form_read_at_arm_true_with_row_authority(monkeypatch):
    from types import SimpleNamespace

    from flliper.srt.pdflip import prefill_oom_retreat as POR

    monkeypatch.setattr(POR, "followers_execute_row", lambda sched: True)
    tree = SimpleNamespace()
    assert LBV.arm(SimpleNamespace(ps=SimpleNamespace(pp_size=3, pp_rank=0), tree_cache=tree))
    assert LBV.state(tree).row_form is True and LBV.same_pass_applies(tree) is True


def test_default_on():
    assert LBV.same_pass_enabled() is True
