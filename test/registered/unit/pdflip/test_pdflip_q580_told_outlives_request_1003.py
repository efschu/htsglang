# SPDX-License-Identifier: Apache-2.0
"""Q-580 (27B dual y8r, boot dkr27bnvfp4dual1mpsleepbar1fs10030958, 55c95a89c7):
the told outlived its request.

METAL (P log, rid pdflip-0-6 = 25 tokens, FORK_ANCHOR cut 25->18):

  10:03:36  instance 1 (leg 1) served; the front re-routes it (spec 3.6, uncached 7 > X=1)
  10:04:20  instance 2: PP0 ``PDFLIP P-FORK-CUT TOLD rid=pdflip-0-6 fork=16`` -- the told is
            published (told_map[rid] on PP0, absorbed on the followers), the request stays
            QUEUED (pdflip-0-9's chunk holds the pass)
  10:04:21  front ``DUAL P-PAUSE`` -> /abort_request: instance 2 leaves every queue unadmitted.
            PP0's adder never visited it (told_map[rid] stands), the followers did (H91 KEPT
            consumed theirs); the abort dropped the request and nothing else
  10:04:22  instance 3: PP0 admits it IN ITS INTAKE PASS on the stale told (``#969 EXTENT
            ('pdflip-0-6', 0, 18)`` right after ``#915 PREFETCH REFUSED``, before any publish),
            its own told is never put on the wire; PP1 queues it without a told and defers
            PP0's frame (#791T) until ``PpRowDeferCapExceeded: #791T STORE-TOLD HOP OVERDUE``
            -> #1223 DEBUG-HOLD -> W17.

THE TESTS drive the REAL abort (``Scheduler._abort_request_now``), the REAL PP0 intake
(``pdflip_store_told.intake``) and the REAL admission gate (``p_intake.told_admission`` /
``told_pending``) on stand-in schedulers. RED on edeb022aee: PP0 admits instance 3 on
instance 2's told; a follower that had not consumed its told would register instance 3
with it. GREEN: the told dies with its request on every rank, and PP0's intake drops a
told of an earlier instance by name.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import scheduler as SC  # noqa: E402
from flliper.srt.managers import pdflip_store_told as ST  # noqa: E402
from flliper.srt.pdflip import p_intake  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

RID = "pdflip-0-6"


class _Tree:
    """The store-read surface ``admission`` touches: every read terminated,
    nothing loaded (pdflip-0-6 is 'too_short' on every instance)."""

    def __init__(self):
        self.prefetch_loaded_tokens_by_reqid = {}
        self.released = []

    def check_prefetch_progress(self, rid):
        return True

    def completed_prefetch_tokens(self, rid):
        return 0

    def pop_prefetch_loaded_tokens(self, rid):
        return 0

    def release_aborted_request(self, rid):
        self.released.append(rid)


class _Sched:
    """A told-ARMED P rank, with what ``_abort_request_now`` reads."""

    def __init__(self, pp_rank: int):
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self._pdflip_store_told_armed = True
        self._pdflip_store_told = {}
        self._pdflip_store_held = {}
        self._pdflip_store_fork = {}
        self._pdflip_told_paced_on = False
        self._pdflip_told_fallback_on = False
        self.tree_cache = _Tree()
        self.waiting_queue = []
        self.chunked_req = None
        self.anchor_tails = ()
        self._pdflip_intake_watch = None
        self.enable_hicache_storage = True
        self.disaggregation_mode = SC.DisaggregationMode.NULL
        self.sent = []
        self.ipc_channels = types.SimpleNamespace(
            send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r=None: self.sent.append(o))
        )
        self.grammar_manager = types.SimpleNamespace(abort_requests=lambda r: None)
        self.running_mbs, self.mbs = [], []
        self.kv_session_offload = None
        self.page_size = 1
        self.registered = []

    # the parts of the scheduler the abort calls that are not under test
    def _pdflip_abort_dormant_hold(self, recv_req):
        return None

    def _pdflip_d_park_abort(self, recv_req):
        return None

    def _prefetch_kvcache(self, req, limit_tokens=None):
        self.registered.append((req.rid, limit_tokens))
        return "declined:too_short"

    def abort(self, rid):
        if self.ps.pp_rank > 0:
            # #1180-W released: PP0's forwarded schedule decided it (the same path)
            self.__dict__.setdefault("_pdflip_force_waiting_abort", set()).add(rid)
        SC.Scheduler._abort_request_now(self, SC.AbortReq(rid=rid))


def _req(n: int = 25):
    return types.SimpleNamespace(
        rid=RID, origin_input_ids=list(range(n)), mamba_pool_idx=None, _pdflip_early_told=None
    )


def _instance2_published_and_queued(pp_rank: int):
    """PP0 published instance 2's told (fork 16 rides it); the request is queued."""
    s = _Sched(pp_rank)
    req2 = _req()
    s.waiting_queue = [req2]
    s._pdflip_store_told[RID] = 0
    s._pdflip_store_fork[RID] = 16
    return s, req2


def _admit(s, req):
    skips = []
    credit = p_intake.told_admission(s, req, lambda k, r: skips.append((k, r)), ST.admission)
    return credit, skips


def test_pp0_abort_takes_the_published_told_with_it():
    """RED on edeb022aee: told_map[pdflip-0-6] survives the abort, and instance 3
    is admitted on it in PP0's intake pass -- before its own told exists."""
    s, req2 = _instance2_published_and_queued(pp_rank=0)
    s.abort(RID)
    assert s.waiting_queue == [] and s.tree_cache.released == [RID]
    assert RID not in s._pdflip_store_told
    assert RID not in s._pdflip_store_fork
    req3 = _req()
    s.waiting_queue = [req3]
    credit, skips = _admit(s, req3)
    assert credit is None, "PP0 admitted a new instance on an earlier instance's told"
    assert skips == [(ST.SKIP_TOLD_PENDING, RID)]


def test_pp0_intake_drops_a_told_of_an_earlier_instance(caplog):
    """Defence in depth (any exit path the abort hook does not see): a told
    standing at PP0's intake was published for an earlier instance."""
    s, _ = _instance2_published_and_queued(pp_rank=0)
    s.waiting_queue = []                         # instance 2 left by some other path
    req3 = _req()
    with caplog.at_level("WARNING"):
        verdict = ST.intake(s, req3, lambda g: None)
    s.waiting_queue = [req3]
    assert verdict == "declined:too_short"
    assert s._pdflip_store_held[RID] is req3      # held -> PP0 publishes ITS told next pass
    assert RID not in s._pdflip_store_told and RID not in s._pdflip_store_fork
    assert "Q-580 STALE-TOLD rid=pdflip-0-6" in caplog.text
    credit, _ = _admit(s, req3)
    assert credit is None                         # admitted only after its own publish


def test_pp0_intake_without_a_stale_told_is_unchanged(caplog):
    s = _Sched(0)
    req = _req()
    with caplog.at_level("WARNING"):
        assert ST.intake(s, req, lambda g: None) == "declined:too_short"
    assert s._pdflip_store_held[RID] is req and s.registered == [(RID, None)]
    assert "Q-580" not in caplog.text


def test_follower_abort_takes_an_unconsumed_told_with_it():
    """The mirror image (a follower whose adder did NOT visit instance 2): the
    stale told would register instance 3 with instance 2's span at intake and
    let #791T pass a frame whose told is not this instance's. RED on edeb022aee."""
    s, req2 = _instance2_published_and_queued(pp_rank=1)
    s._pdflip_store_told_satisfied = {RID: 0}
    s.abort(RID)
    assert s.waiting_queue == []
    assert RID not in s._pdflip_store_told
    assert RID not in s._pdflip_store_fork
    assert RID not in s._pdflip_store_told_satisfied
    req3 = _req()
    assert ST.intake(s, req3, lambda g: None) == "declined:%s" % ST.GATE_HELD
    s.waiting_queue = [req3]
    assert s._pdflip_store_held[RID] is req3        # registers when ITS told arrives
    assert p_intake.told_pending(s, req3) is True  # #791T defers until then


def test_follower_consumed_told_is_unchanged_by_the_abort():
    """The metal follower: its visit consumed the told (H91 KEPT); the abort
    leaves nothing behind, as before (settle_told drops the kept verdict)."""
    s, req2 = _instance2_published_and_queued(pp_rank=1)
    credit, _ = _admit(s, req2)
    assert credit == 0 and RID not in s._pdflip_store_told
    assert p_intake._kept(s)[RID].req is req2
    s.abort(RID)
    p_intake.settle_told(s, s.waiting_queue)
    req3 = _req()
    s.waiting_queue = [req3]
    assert p_intake.told_pending(s, req3) is True


def test_the_new_instances_own_told_admits_on_every_rank():
    """After the fix the ranks agree again: PP0 publishes instance 3's told, the
    follower absorbs it, both admit on it."""
    pp0, _ = _instance2_published_and_queued(pp_rank=0)
    pp1, _ = _instance2_published_and_queued(pp_rank=1)
    for s in (pp0, pp1):
        s.abort(RID)
    req3_0, req3_1 = _req(), _req()
    ST.intake(pp0, req3_0, lambda g: None)
    ST.intake(pp1, req3_1, lambda g: None)
    pp0.waiting_queue, pp1.waiting_queue = [req3_0], [req3_1]
    assert _admit(pp0, req3_0)[0] is None and p_intake.told_pending(pp1, req3_1)
    # PP0's publish (single-phase told=0, as pdflip-0-6 on metal)
    pp0._pdflip_store_told[RID] = 0
    pp0._pdflip_store_held.pop(RID)
    rest = ST._follower_absorb_impl(pp1, [ST.PdFlipStoreTold(rid=RID, told=0)])
    assert rest == []
    assert p_intake.told_pending(pp1, req3_1) is False
    assert _admit(pp0, req3_0)[0] == 0
    assert _admit(pp1, req3_1)[0] == 0


def test_replay_10031032_no_fork_anchor_pdflip_0_18():
    """Second death, WITHOUT FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN (boot
    dkr27bnvfp4dual1mpsleepbar1fs10031032, probe step 'load long2'; the log's
    'pdflip-0-1' is rid[:8] -- the #1223 dump names pdflip-0-18):

      10:36:33 PP0 intake pdflip-0-18 (10585 tokens), its told=0 goes on the wire
               (P-FORK-CUT TOLD fork=59 src=store) in the SAME pass the front's
               DUAL P-PAUSE abort lands (CTRL-FWD n_wire=5: 3 AbortReq + told + Admit)
      followers: told absorbed, abort held (#1180-W), admission visit (witness
               n=38..41) consumes the told, WAITING-ABORT pop at 10:36:34
      10:36:35 instance 2 (P-PAUSED requeue); PP0 registers its read (12.3 s)
      10:36:47 PP0 '#1400 STORE-TOLD WAITED told=0' -> #969 EXTENT (0, 1024):
               admitted on instance 1's told, its own never published
      10:36:48 PP1 #791T x4 -> PpRowDeferCapExceeded -> W17.
    RED on edeb022aee (PP0 admits instance 2), GREEN with Q-580."""
    rid = "pdflip-0-18"
    pp0, pp1 = _Sched(0), _Sched(1)
    r0, r1 = _req(10585), _req(10585)
    r0.rid = r1.rid = rid
    pp0.waiting_queue, pp1.waiting_queue = [r0], [r1]
    # PP0's publish of instance 1 (told 0, fork 59) ...
    pp0._pdflip_store_told[rid] = 0
    pp0._pdflip_store_fork[rid] = 59
    # ... and the P-PAUSE abort in the same pass: PP0 applies it at receipt
    pp0.abort(rid)
    # the follower absorbs the told off the same list, then visits it once
    assert ST._follower_absorb_impl(pp1, [ST.PdFlipStoreTold(rid=rid, told=0)]) == []
    assert _admit(pp1, r1)[0] == 0                     # H91 KEPT took it
    pp1.abort(rid)                                     # #1180-W pop
    p_intake.settle_told(pp1, pp1.waiting_queue)
    # instance 2
    n0, n1 = _req(10585), _req(10585)
    n0.rid = n1.rid = rid
    ST.intake(pp0, n0, lambda g: None)
    ST.intake(pp1, n1, lambda g: None)
    pp0.waiting_queue, pp1.waiting_queue = [n0], [n1]
    assert _admit(pp0, n0)[0] is None, "PP0 admitted instance 2 on instance 1's told (STORE-TOLD WAITED told=0)"
    assert p_intake.told_pending(pp1, n1) is True     # and the follower waits for the SAME told


def test_paced_window_of_the_aborted_request_is_dropped():
    s = _Sched(0)
    s._pdflip_told_paced_on = True
    req2 = _req()
    s.waiting_queue = [req2]
    ST._pacing(s)[RID] = ST._Pace(req=req2, told=4096, published_at=0.0, published_pass=1,
                                  window_s=1.0, absolute=False)
    s.abort(RID)
    assert RID not in ST._pacing(s)              # never an Admit for instance 3 from it


def test_not_armed_is_a_no_op():
    s = types.SimpleNamespace(waiting_queue=[], ps=types.SimpleNamespace(pp_rank=0))
    req = _req()
    assert ST.forget_left_queue(s, req, "abort") == []
    assert not hasattr(s, "_pdflip_store_told_armed")
    s._pdflip_store_told_armed = False
    s._pdflip_store_told = {RID: 0}
    assert ST.forget_left_queue(s, req, "abort") == []
    assert s._pdflip_store_told == {RID: 0}


def test_another_rid_is_untouched():
    s, _ = _instance2_published_and_queued(pp_rank=0)
    other = types.SimpleNamespace(rid="pdflip-0-9", origin_input_ids=[1], mamba_pool_idx=None)
    s._pdflip_store_told["pdflip-0-9"] = 4096
    s.waiting_queue.append(other)
    s.abort(RID)
    assert s._pdflip_store_told == {"pdflip-0-9": 4096}
    assert s.waiting_queue == [other]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
