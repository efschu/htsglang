"""Review P-COMMIT (8dc89e8dff, 21.10.2026): gaps of test_pcommit_vision_at_admit_1021.py, written by the reviewer.

The author's suite proves the PP0 half (the verdict is formed beside the Admit) and a follower that calls
``vision_verdict.absorb`` BY HAND. What it does not drive is the real wire between the two, nor PP0's own vision pass in
the same pass, nor the Admit variants other than the plain paced one. Pinned here (hermetic, CPU):

  R1. the REAL wire: PP0's post-publish stamp (flush verdicts to the FRONT) keeps the Admit and the verdict at the tail in
      order; a follower's REAL ``pdflip_store_told.follower_absorb`` + ``flush_verdict.follower_absorb`` take the Admit off
      and leave the verdict for dispatch; PP1 (absorbs) and PP2 (gets the identical relayed list) end in the SAME state
      -- told registered, request released, nothing held by ``follower_pass`` in this pass;
  R2. PP0's own ``vision_rank_pass`` in the same pass: the request is SKIP-marked (the admission belt
      ``skip_still_covered`` arms), not parked, and no second verdict is queued for the next intake;
  R3. the other Admit forms: H64 GROUP-FLOOR below / at image_end, an S64 raise, a TW twin single-phase told -- the
      verdict follows PP0's RECORDED told (``_pdflip_store_told``), never the wire object's number alone;
  R4. a raise inside the injected SKIP rule: no verdict, no ``sent``/``released`` mark, the plan's vision pass queues it
      as before (one pass later, never never);
  R5. HARDENING (xfail strict, review note H1): PP0 must never keep a rid released whose verdict did not reach the wire;
      today ``pp0_verdicts_at_admit`` absorbs on PP0 BEFORE the caller appended the verdict, so a raise after the
      absorb (only the log call follows it -- practically unreachable) would split PP0 from the followers;
  R6. #PDRAIN: an entry whose after-commit never ran (the receive raised) never produces a stale line on a later
      non-blocking pass, and a later blocking pass logs its own numbers.
"""

import logging
import os
import sys
import time as _time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_store_told as told  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402
from flliper.srt.pdflip import flush_verdict as fv  # noqa: E402
from flliper.srt.pdflip import vision_rank_runner as vrr  # noqa: E402
from flliper.srt.pdflip import vision_verdict as vv  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

import test_pcommit_vision_at_admit_1021 as base  # noqa: E402
from test_pcommit_vision_at_admit_1021 import _pf  # noqa: E402,F401  (autouse: PF answers as on the metal)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

RID, TOLD, IMAGE_END = base.RID, base.TOLD, base.IMAGE_END


def _follower(rank: int):
    f = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=rank, pp_size=3, tp_size=1),
        waiting_queue=[base._vision_req()], chunked_req=None, pdflip_dormant=False,
        _pdflip_store_told_armed=True, _pdflip_store_told={}, _pdflip_store_held={},
        _pdflip_fb_follower=None,
    )
    vv.arm(f)
    return f


def _dispatch(f, rest):
    """process_input_requests' part that matters here: the H125f handler (scheduler.handle_pdflip_vision_verdict)."""
    for item in rest:
        if isinstance(item, vv.PdFlipVisionVerdict):
            vv.absorb(f, item)


def _follower_pass_of(f, wire):
    """scheduler_pp_mixin, follower branch, in its order: relay (the list as received), flush verdicts, store told,
    dispatch."""
    relayed = list(wire)
    rest = fv.follower_absorb(f, list(wire), lambda *a: None)
    rest = told.follower_absorb(f, rest)
    _dispatch(f, rest)
    return relayed, rest


# ---------------------------------------------------------------------------
# R1. the real wire, PP0 -> PP1 -> PP2


def test_r1_real_wire_pp0_stamp_and_both_followers_release_in_the_same_pass():
    req = base._vision_req()
    s = base._sched(req)
    abort = SimpleNamespace(rid="pdflip-other-1")            # a request object of this pass's intake
    wire = told.pp0_publish(s, [abort])
    # PP0's flush-verdict stamp puts last pass's verdicts at the FRONT (scheduler_pp_mixin, after pp0_publish)
    s._pdflip_flush_verdict_armed = True
    fv._state(s)["out"].append(fv.PdFlipFlushVerdict(seq=7, passed=True, detail=""))
    wire = fv.pp0_wire(s, wire)
    kinds = [type(x).__name__ for x in wire]
    assert kinds == ["PdFlipFlushVerdict", "SimpleNamespace", "PdFlipStoreAdmit", "PdFlipVisionVerdict"], kinds

    pp1 = _follower(1)
    relayed, rest = _follower_pass_of(pp1, wire)
    assert [type(x).__name__ for x in rest] == ["SimpleNamespace", "PdFlipVisionVerdict"]   # Admit absorbed, verdict left
    pp2 = _follower(2)
    _follower_pass_of(pp2, relayed)                          # PP2 gets PP1's relay = the identical list

    for f in (pp1, pp2):
        assert f._pdflip_store_told[RID] == TOLD             # the Admit's told is this rank's admission verdict
        assert RID in f._pdflip_vision_released
        assert vv.follower_pass(f) == []                     # nothing held: admitted in THIS pass, as on PP0
    assert RID in s._pdflip_vision_released and vv.gate_held(s, s.waiting_queue) == []


# ---------------------------------------------------------------------------
# R2. PP0's own vision pass in the Admit's pass


def test_r2_pp0_vision_pass_skips_marks_and_queues_no_second_verdict():
    req = base._vision_req()
    s = base._sched(req)
    s._pdflip_vision_inflight = None
    s._pdflip_vision_arm_refusal = None
    told.pp0_publish(s, [])
    parked = vrr.vision_rank_pass(s)
    assert parked == []                                      # PP0 admits it in this pass (base: held one pass)
    assert getattr(req, "_pdflip_vision_skip", False) is True  # skip_still_covered guards PP0's admission
    assert s._pdflip_vision_origin_verdicts == []            # no duplicate verdict at the next intake
    assert s.waiting_queue == [req]


def test_r2_reference_without_the_hook_pp0_parks_the_request():
    req = base._vision_req()
    s = base._sched(req)
    s._pdflip_vision_inflight = None
    s._pdflip_vision_arm_refusal = None
    # what 50def7d4c8 did: the paced publish alone, then the plan's vision pass
    told._pp0_publish_paced(s, [])
    parked = vrr.vision_rank_pass(s)
    assert [r.rid for _, r in parked] == [RID]
    assert s._pdflip_vision_origin_verdicts == [RID]


# ---------------------------------------------------------------------------
# R3. the other Admit forms -- PP0's recorded told decides


@pytest.mark.parametrize("value,want", [(IMAGE_END - 1, []), (IMAGE_END, [RID])])
def test_r3_group_floor_admit(value, want):
    req = base._vision_req()
    s = base._sched(req)
    s._pdflip_told_pacing = {}
    s._pdflip_store_told[RID] = value                        # H64: told_map[rid] = told_final at the floor
    admit = told.PdFlipStoreAdmit(rid=RID, told=value)
    setattr(admit, fb.WIRE_FLOOR, 1)
    wire = told._vision_at_admit(s, [], [admit])
    assert [v.rid for v in base._verdicts(wire)] == want


def test_r3_raise_admit_covers():
    from flliper.srt.managers import pdflip_told_raise as traise

    req = base._vision_req()
    s = base._sched(req, told_value=IMAGE_END - 500)
    s._pdflip_told_pacing = {}
    s._pdflip_store_told[RID] = TOLD                         # S64: raised above the read-ahead, past the image
    admit = told.PdFlipStoreAdmit(rid=RID, told=TOLD)
    setattr(admit, traise.WIRE_RAISE, 1)
    assert [v.rid for v in base._verdicts(told._vision_at_admit(s, [], [admit]))] == [RID]


def test_r3_wire_number_alone_never_releases():
    """PP0 has no recorded told for the rid (nothing paced, nothing held): the object's told is not a verdict."""
    req = base._vision_req()
    s = base._sched(req)
    s._pdflip_told_pacing = {}
    wire = told._vision_at_admit(s, [], [told.PdFlipStoreAdmit(rid=RID, told=TOLD)])
    assert base._verdicts(wire) == [] and RID not in s._pdflip_vision_released


def test_r3_twin_single_phase_told_is_an_admission():
    req = base._vision_req()
    s = base._sched(req)
    s._pdflip_told_pacing = {}
    s._pdflip_store_told[RID] = TOLD
    twin = told.PdFlipStoreToldTwin(rid=RID, told=TOLD, absolute=True)
    assert [v.rid for v in base._verdicts(told._vision_at_admit(s, [], [twin]))] == [RID]


# ---------------------------------------------------------------------------
# R4. the SKIP rule raises


def test_r4_skip_rule_raises_old_path_takes_over():
    req = base._vision_req()
    s = base._sched(req)
    s._pdflip_told_pacing = {}
    s._pdflip_store_told[RID] = TOLD

    def boom(scheduler, r):
        raise RuntimeError("tensor class")

    assert vv.pp0_verdicts_at_admit(s, [RID], boom) == []
    assert RID not in s._pdflip_vision_released and RID not in s._pdflip_vision_verdict_sent
    vrr.skip_cached(s, [req])
    vv.queue_verdicts(s, s.waiting_queue, vrr.pending_items, s._pdflip_vision_refused)
    assert s._pdflip_vision_origin_verdicts == [RID]         # one pass later, as before the fix


# ---------------------------------------------------------------------------
# R5. hardening H1 (not a live defect: only the log call follows the absorb)


@pytest.mark.xfail(strict=True, reason="review H1: absorb on PP0 precedes the wire append; a raise in between "
                                       "leaves PP0 released and the followers without a verdict")
def test_r5_pp0_released_implies_verdict_on_the_wire(monkeypatch):
    req = base._vision_req()
    s = base._sched(req)
    s._pdflip_told_pacing = {}
    s._pdflip_store_told[RID] = TOLD

    class _Raising:
        def info(self, msg, *a, **k):
            if str(msg).startswith("%s rid=%s covered="):
                raise RuntimeError("raise after the absorb")

        def __getattr__(self, name):
            return getattr(logging.getLogger(vv.__name__), name)

    monkeypatch.setattr(vv, "logger", _Raising())
    wire = told._vision_at_admit(s, [], [told.PdFlipStoreAdmit(rid=RID, told=TOLD)])
    on_wire = RID in [v.rid for v in base._verdicts(wire)]
    assert (RID in s._pdflip_vision_released) == on_wire


# ---------------------------------------------------------------------------
# R6. #PDRAIN entry hygiene


def test_r6_pdrain_no_stale_line_after_a_raised_receive(caplog):
    s = base._sched(base._text_req("r1"))
    t, pending = told.s82_output_pending_at_entry(s)        # blocking: entry snapshot taken
    assert pending is True and s._pdrain_entry is not None
    # the receive raised: s82_after_commit never ran for it. Next pass: the slot is free, nothing blocks.
    s._s82_slot = (0, 1)
    t2, p2 = told.s82_output_pending_at_entry(s)
    assert p2 is False
    with caplog.at_level(logging.INFO, logger=told.logger.name):
        told.s82_after_commit(s, 1, (t2, p2))
    assert not [r for r in caplog.records if r.getMessage().startswith("#PDRAIN")]
    # a later blocking pass logs its OWN entry (fresh snapshot, fresh wait)
    s._s82_slot = (2, 0)
    s._s82_launch_rec = {0: (_time.monotonic() - 1.0, 100, 200)}
    s.waiting_queue = []
    t3, p3 = told.s82_output_pending_at_entry(s)
    with caplog.at_level(logging.INFO, logger=told.logger.name):
        told.s82_after_commit(s, 0, (t3, p3))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("#PDRAIN")]
    assert len(lines) == 1 and "cause=idle" in lines[0] and "new=100 cached=200" in lines[0]
    assert getattr(s, "_pdrain_pause_n", 0) == 0             # idle is not a P pause
