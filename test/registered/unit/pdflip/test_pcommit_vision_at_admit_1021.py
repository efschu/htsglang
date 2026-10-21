"""P-COMMIT (21.10.2026): what PP0's ``output_commit`` is, and the vision verdict that waited behind it.

Metal (NF rc30, boot ..._1021_063252 = 1d822bd150, same code as 50def7d4c8 in every file touched here; P = PP3):

* output_commit (``#PGAP pp_rank=0 ... host[... output_commit=...]``) is the span around
  ``_pp_commit_send_output_work_and_preprocess_output_tensors`` (scheduler_pp_mixin ``with _pov.span("output_commit")``)
  -- on PP0 the blocking receive of the older set's output from PP2. All 166 PP0 forwards with output_commit > 500 ms:
  output_commit - (PP1 gpu_fwd_ms + PP2 gpu_fwd_ms of the older set) = median 210 ms. fwd217: 3987 = 2556.5 + 1289.8
  + 141; fwd220: 4138 = 2676.1 + 1324.9 + 137. The span is the downstream drain, not host work.
* fwd220 (09:37:12-16, pdflip-144-1508, an image request): ``S82 COMMIT-PARK outcome=acks`` -> ``S82 PARK-ADMIT``
  (told=113728) -> plan: ``W102 PdFlipVisionStage SKIP reason=image-cached covered=113728 src=told
  image_end=113719`` -> ``W102c PdFlipVisionVerdictGate VERDICT ... (on the wire at the next intake)`` -> the next
  intake came after ``#1466 PASS-STALL ... commit_ms=4138``; ``#PGAP fwd=220 gpu_gap_ms=4492.6``.

Pinned (hermetic, CPU):
  1. METAL fwd220: the told-covered image's verdict rides the Admit's chain and PP0 releases it in this pass
     (RED on 50def7d4c8: the chain carries the Admit only);
  2. a told short of the image, a text request, a refused or already released rid: no verdict at admit -- the
     plan's vision pass decides at the next intake as before;
  3. the single-phase (unpaced) told is an admission too;
  4. rank-uniform: a follower that dispatches the same chain releases the request in the SAME pass, and PP0's
     plan queues no second verdict;
  5. #PDRAIN: a blocking PP0 receive names who waited behind it (idle = nobody); fwd217's numbers;
  6. the wiring: both return points of pp0_publish go through the verdict hook; the instrument rides S82's
     entry/after-commit pair.
"""

import inspect
import logging
import os
import sys
import time as _time
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_store_told as told  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402
from flliper.srt.managers import pdflip_told_fidelity as tf  # noqa: E402
from flliper.srt.managers import pdflip_told_raise as traise  # noqa: E402
from flliper.srt.pdflip import pp0_commit_park as cp  # noqa: E402
from flliper.srt.pdflip import vision_rank_runner as vrr  # noqa: E402
from flliper.srt.pdflip import vision_verdict as vv  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

RID = "pdflip-144-1508"          # metal 09:37:12 PP0
TOLD = 113728                    # 'W102 PdFlipVisionStage SKIP rid=pdflip-144-1508 ... covered=113728 src=told'
IMAGE_END = 113719               # '... image_end=113719'
PROMPT = 116764                  # '#969 EXTENT ... (pdflip-144-1508, 113728, 116764, 113728, 3036)'


class _Item:
    def __init__(self, start, end):
        self.feature = torch.zeros(4, 8)
        self.precomputed_embeddings = None
        self.modality = "image"
        self.offsets = [(start, end)]


def _vision_req(rid=RID, end=IMAGE_END):
    return SimpleNamespace(rid=rid, origin_input_ids=list(range(PROMPT)), extra_key=None,
                           multimodal_inputs=SimpleNamespace(mm_items=[_Item(end - 1200, end - 1)]),
                           prefix_indices=None, host_hit_length=0)


def _text_req(rid):
    return SimpleNamespace(rid=rid, origin_input_ids=list(range(PROMPT)), extra_key=None,
                           multimodal_inputs=None, prefix_indices=None, host_hit_length=0)


class _Ctr:
    """pp_flip_counters with the per-kind axis: the older set's output owed by PP2, not posted yet."""

    def __init__(self):
        self.sent_n, self.att_n, self.cons_n = 4, 4, 4

    def sent(self, chan, rank):
        return self.sent_n

    def kind_sent_total(self, chan, rank):
        return self.sent_n

    def sent_of_kind(self, chan, kind, rank):
        return self.sent_n if kind == "output" else 0

    def attempted_of_kind(self, chan, kind, rank):
        return self.att_n if kind == "output" else 0

    def local_consumed_of_kind(self, chan, kind):
        return self.cons_n if kind == "output" else 0


def _older_batch():
    # slot 0: pdflip-144-1507, 16235 new on 18752 cached ('Prefill rank batch, #new-token: 16235, #cached-token: 18752')
    return SimpleNamespace(reqs=[SimpleNamespace(rid="pdflip-144-1507")],
                           forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
                           contains_last_prefill_chunk=True, return_logprob=False)


def _sched(req, told_value=TOLD):
    now = told._clock()
    s = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1),
        waiting_queue=[req], chunked_req=None, pdflip_dormant=False,
        _pdflip_store_told_armed=True,
        _pdflip_store_held={}, _pdflip_store_told={},
        _pdflip_told_paced_on=True, _pdflip_told_fallback_on=True, _pdflip_told_pass_n=5,
        _pdflip_told_pacing={req.rid: told._Pace(req=req, told=told_value, published_at=now - 0.05,
                                                 published_pass=4, window_s=5.0, absolute=True)},
        _pdflip_fb_open={req.rid: fb._Open(told=told_value, deadline=now + 30.0, published_pass=4,
                                           published_at=now - 0.05)},
        tree_cache=SimpleNamespace(ongoing_prefetch={}),
        mbs=[_older_batch(), None, None], _s82_slot=(2, 0),     # metal 'S82 COMMIT-PARK ... slot=(2, 0)'
        server_args=SimpleNamespace(pp_async_batch_depth=0),
        pp_flip_counters=_Ctr(), pp_group=SimpleNamespace(rank_in_group=0, world_size=3),
        _pp_flip_upstream=lambda: 2,
        _s82_launch_rec={0: (_time.monotonic() - 0.1, 16235, 18752)},
        _pdflip_vision_refused=set(),
    )
    vv.arm(s)                     # PP0's half of the H125f gate (arm_rank_stage does this on the metal)
    return s


@pytest.fixture(autouse=True)
def _pf(monkeypatch):
    """The PF/paced pieces that are not under test answer as on the metal: the acks of both followers are in
    (metal: 'S82 COMMIT-PARK n=92 outcome=acks rid=pdflip-144-1508 waited_ms=5.3')."""
    cp.reset()
    monkeypatch.delenv(cp.ENV, raising=False)
    monkeypatch.setattr(fb, "pp0_has_stream", lambda s: True)
    monkeypatch.setattr(fb, "_frist_verdict", lambda *a, **k: None)
    monkeypatch.setattr(traise, "pp0_raise_step", lambda *a, **k: None)
    monkeypatch.setattr(tf, "pp0_verdict", lambda s, req, t, absolute: (t, t))

    def harvest(s, now=None):
        for rid, o in fb._pp0_open_map(s).items():
            o.acks.update({1: o.told, 2: o.told})
        return 0

    monkeypatch.setattr(fb, "pp0_harvest", harvest)
    yield


def _admits(out):
    return [o for o in out if isinstance(o, told.PdFlipStoreAdmit)]


def _verdicts(out):
    return [o for o in out if isinstance(o, vv.PdFlipVisionVerdict)]


# ---------------------------------------------------------------------------
# 1. metal fwd220


def test_metal_0937_told_covered_image_verdict_rides_the_admit_chain():
    req = _vision_req()
    s = _sched(req)
    out = told.pp0_publish(s, [])
    assert [a.rid for a in _admits(out)] == [RID] and _admits(out)[0].told == TOLD
    # RED on 50def7d4c8: the chain carried the Admit only, the verdict waited for the next intake (4138 ms later)
    assert [v.rid for v in _verdicts(out)] == [RID]
    assert out.index(_admits(out)[0]) < out.index(_verdicts(out)[0])      # behind its Admit on the wire
    assert RID in s._pdflip_vision_released and RID in s._pdflip_vision_verdict_sent
    assert vv.gate_held(s, s.waiting_queue) == []                          # PP0 admits it in THIS pass
    assert s._pdflip_vision_origin_verdicts == []                          # nothing left for the next intake


# ---------------------------------------------------------------------------
# 2. the cases that keep the old path


@pytest.mark.parametrize("case", ["told_short", "text", "refused", "released", "gate_off", "dormant"])
def test_no_verdict_at_admit_where_the_plan_still_decides(case):
    req = _text_req(RID) if case == "text" else _vision_req(end=TOLD + 64 if case == "told_short" else IMAGE_END)
    s = _sched(req)
    if case == "refused":
        s._pdflip_vision_refused.add(RID)
    elif case == "released":
        s._pdflip_vision_released.add(RID)
    elif case == "gate_off":
        s._pdflip_vision_gate = False
    out = told.pp0_publish(s, [])
    if case == "dormant":
        # the Admit itself is formed by the paced branch; the gate stands down while P sleeps
        s2 = _sched(_vision_req())
        s2.pdflip_dormant = True
        assert vv.pp0_verdicts_at_admit(s2, [RID], vrr.stage_skip_verdict) == []
        return
    assert [a.rid for a in _admits(out)] == [RID]
    assert _verdicts(out) == []
    if case == "told_short":
        # the plan's vision pass stages it (tower) and queues the verdict as before
        v, covered, src = vrr.stage_skip_verdict(s, req)
        assert (v, covered, src) == (vrr.STAGE, TOLD, "told")


def test_a_read_ahead_is_no_admission():
    """Only an admission releases: a paced read-ahead (membership still open) never carries the verdict."""
    req = _vision_req()
    s = _sched(req)
    ahead = told.PdFlipStoreTold(rid=RID, told=TOLD, paced=True)
    s._pdflip_store_told[RID] = TOLD
    wire = told._vision_at_admit(s, [], [ahead])
    assert _verdicts(wire) == []


# ---------------------------------------------------------------------------
# 3. single-phase told


def test_single_phase_told_is_an_admission():
    req = _vision_req()
    s = _sched(req)
    s._pdflip_store_told[RID] = TOLD
    wire = told._vision_at_admit(s, [], [told.PdFlipStoreTold(rid=RID, told=TOLD)])
    assert [v.rid for v in _verdicts(wire)] == [RID]
    assert told._vision_at_admit(s, [], [told.PdFlipStoreAdmit(rid="x", told=0)]) == \
        [told.PdFlipStoreAdmit(rid="x", told=0)]                           # told=0 fallback: no image covered
    recv = []
    assert told._vision_at_admit(s, recv, recv) is recv                     # nothing published: the same list


# ---------------------------------------------------------------------------
# 4. rank-uniform release


def test_follower_releases_in_the_pass_that_absorbs_the_chain_and_pp0_queues_no_second_verdict():
    req = _vision_req()
    s = _sched(req)
    out = told.pp0_publish(s, [])
    f = SimpleNamespace(waiting_queue=[_vision_req()], pdflip_dormant=False,
                        ps=SimpleNamespace(pp_size=3, pp_rank=1))
    vv.arm(f)
    # the follower dispatches the chain it received: the verdict object is handled like any request
    for item in out:
        if isinstance(item, vv.PdFlipVisionVerdict):
            vv.absorb(f, item)
    assert vv.follower_pass(f) == []                       # admitted in THIS pass on PP1 as on PP0
    # PP0's plan: the vision pass would SKIP (told covers) and must not put a second verdict on the wire
    vv.queue_verdicts(s, s.waiting_queue, vrr.pending_items, s._pdflip_vision_refused)
    assert s._pdflip_vision_origin_verdicts == []


def test_base_behaviour_without_the_hook_is_one_pass_late():
    """The pre-fix flow, kept as the reference: the plan's vision pass queues the verdict for the next intake and
    holds the request on PP0 in the Admit's pass."""
    req = _vision_req()
    s = _sched(req)
    s._pdflip_store_told[RID] = TOLD
    stage, wait = vrr.skip_cached(s, [req])
    assert stage == [] and wait == [] and req._pdflip_vision_skip
    vv.queue_verdicts(s, s.waiting_queue, vrr.pending_items, s._pdflip_vision_refused)
    assert s._pdflip_vision_origin_verdicts == [RID]
    assert vv.gate_held(s, s.waiting_queue) == [req]       # held on PP0 until the next intake


# ---------------------------------------------------------------------------
# 5. #PDRAIN


def test_pdrain_names_who_waited_behind_the_receive(caplog):
    # fwd217 (09:33:20-25): pdflip-141-1469 paced behind the 15019-token set, commit_ms=3987
    req = _text_req("pdflip-141-1469")
    s = _sched(req)
    s._s82_launch_rec = {0: (_time.monotonic() - 3.987, 15019, 58880)}
    t, pending = told.s82_output_pending_at_entry(s)
    assert pending is True and s._pdrain_entry is not None
    with caplog.at_level(logging.INFO, logger=told.logger.name):
        told.s82_after_commit(s, 0, (t, pending))
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("#PDRAIN")]
    assert len(line) == 1
    assert "cause=paced" in line[0] and "queued=1" in line[0] and "paced=1" in line[0]
    assert "new=15019 cached=58880" in line[0]
    assert s._pdrain_entry is None and s._pdrain_pause_n == 1


@pytest.mark.parametrize("kind,cause", [("empty", "idle"), ("vision", "vision"), ("held_open", "held_open"),
                                        ("held_ready", "held_ready"), ("admissible", "admissible")])
def test_pdrain_classes(kind, cause):
    req = _vision_req() if kind == "vision" else _text_req("r1")
    s = _sched(req)
    s._pdflip_told_pacing = {}
    if kind == "empty":
        s.waiting_queue = []
    elif kind in ("held_open", "held_ready"):
        s._pdflip_store_held = {"r1": req}
        op = SimpleNamespace(host_indices=[1])
        s.tree_cache = SimpleNamespace(ongoing_prefetch={"r1": SimpleNamespace(operation=op)},
                                       can_terminate_prefetch=lambda op, tail_hold=None: kind == "held_ready")
    snap = told.pdrain_snapshot(s)
    assert told.pdrain_cause(snap) == cause


def test_pdrain_only_for_a_blocking_receive(caplog):
    s = _sched(_text_req("r1"))
    s._s82_slot = (0, 1)                                    # next slot empty: the receive does not block
    t, pending = told.s82_output_pending_at_entry(s)
    assert pending is False and getattr(s, "_pdrain_entry", None) is None
    with caplog.at_level(logging.INFO, logger=told.logger.name):
        told.s82_after_commit(s, 1, (t, pending))
    assert not [r for r in caplog.records if r.getMessage().startswith("#PDRAIN")]


# ---------------------------------------------------------------------------
# 6. wiring


def test_wiring():
    pub = inspect.getsource(told.pp0_publish)
    assert "_vision_at_admit(scheduler, recv_reqs, _pp0_publish_paced(" in pub
    assert "_vision_at_admit(scheduler, recv_reqs, list(recv_reqs) + out)" in pub
    assert pub.index("_commit_park(") < pub.index("_pp0_publish_paced(")    # S82's order is unchanged
    hook = inspect.getsource(told._vision_at_admit)
    assert "stage_skip_verdict" in hook                     # the plan's own SKIP rule, not a second one
    after = inspect.getsource(told.s82_after_commit)
    assert "_log_pdrain(" in after
    from flliper.srt.managers import scheduler_pp_mixin as m

    src = inspect.getsource(m)
    i = src.index("s82_output_pending_at_entry(self)")
    assert src.index('with _pov.span("output_commit")', i) < src.index("s82_after_commit(self", i)
