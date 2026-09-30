"""HOLD-RELEASE-AFTER-REPLY (30.09., z30y8 epoch 26, P->D wake=16638 ms).

Measured: the #1443 hold release ran INSIDE the kv resume RPC. RPC-STALL-WATCHDOG at +3 s (file
weg2_rpcstall_D_r0_resume_1790792535271.txt) showed TP0 in ``_weg2_release_dormant_hold ->
park_l3.issue_deferred_reads -> prefetch_from_storage`` while TP1/TP2 waited in its
``prefetch_participation_vote`` all_reduce; the RPC returned after 15008 ms, so the flip ended only
then (dc@17270). The hold reads themselves took < 1 s once issued.

Change: the resume marks the release due (every rank: the clear half runs on all ranks or on none),
and ``process_input_requests`` runs it right after that resume's reply was sent -- the same position
of the same broadcast intake list on every rank, before any later request of the list. The flip is
done before the release begins. Instruments: WEG2-HOLD-READ-TIME per rid (pre/bind/alloc/query/
collective/post); the deferred release is watched by the RPC-STALL-WATCHDOG (kind hold_release; its +8 s
second dump is a37762e597's sampler).

DANGER DIRECTION = rank disagreement at the hicache collective (a new wedge): pinned by running the
same intake list through three schedulers.
"""

import collections
import inspect
import logging
import os
import tempfile
import time
import types
from unittest import mock

import pytest

from sglang.srt.managers import scheduler as S
from sglang.srt.managers.scheduler_components import weight_updater as WU
from sglang.srt.weg2 import park_l3 as PL


class _Sent:
    def __init__(self, ev):
        self.ev = ev

    def send_output(self, output, recv_req):
        self.ev.append(("sent", output))


def _sched(ev, hold_n=3, release_raises=False):
    s = object.__new__(S.Scheduler)
    s.session_controller = types.SimpleNamespace(maybe_reap=lambda now: None)
    s.flush_wrapper = types.SimpleNamespace(check_pending=lambda: None)
    s.external_corpus_manager = None
    s.ipc_channels = types.SimpleNamespace(send_to_tokenizer=_Sent(ev), recv_from_rpc=None)
    s.weg2_dormant_hold = [types.SimpleNamespace(rid="h%d" % i) for i in range(hold_n)]
    s.tp_rank = 0

    def dispatch(req):
        ev.append(("disp", req))
        if req == "resume":
            # what the resume's clear half does under the switch
            s._weg2_hold_release_due = True
            return "RESUME-OK"
        return None

    def release():
        ev.append(("release", len(s.weg2_dormant_hold)))
        if release_raises:
            raise RuntimeError("partial release")
        n = len(s.weg2_dormant_hold)
        s.weg2_dormant_hold.clear()
        return n

    s._request_dispatcher = dispatch
    s._weg2_release_dormant_hold = release
    return s


def test_the_release_runs_after_the_resume_reply_and_before_the_next_request():
    ev = []
    s = _sched(ev)
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0"}):
        s.process_input_requests(["gen-a", "resume", "gen-b"])
    assert ev == [("disp", "gen-a"), ("disp", "resume"), ("sent", "RESUME-OK"), ("release", 3),
                  ("disp", "gen-b")]
    assert s._weg2_hold_release_due is False


def test_every_rank_releases_at_the_same_point_of_the_same_list():
    lists = [["resume"], ["gen-a", "resume"], ["resume", "gen-a", "gen-b"], ["gen-a"]]
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0"}):
        for recv in lists:
            seqs = []
            for _rank in range(3):
                ev = []
                _sched(ev).process_input_requests(list(recv))
                seqs.append([e for e in ev if e[0] in ("disp", "release")])
            assert seqs[0] == seqs[1] == seqs[2], recv
            assert (("release", 3) in seqs[0]) == ("resume" in recv)


def test_no_release_without_a_resume_and_only_once():
    ev = []
    s = _sched(ev)
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0"}):
        s.process_input_requests(["gen-a"])
        s.process_input_requests(["resume"])
        s.process_input_requests(["gen-b"])
    assert [e for e in ev if e[0] == "release"] == [("release", 3)]


def test_a_failing_release_propagates_and_is_not_retried():
    ev = []
    s = _sched(ev, release_raises=True)
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0"}):
        with pytest.raises(RuntimeError):
            s.process_input_requests(["resume", "gen-a"])
    assert s._weg2_hold_release_due is False           # never re-run on a later pass
    assert ("sent", "RESUME-OK") in ev                 # the reply had left before


def test_the_deferred_release_is_watched(tmp_path):
    ev = []
    s = _sched(ev)

    def slow():
        time.sleep(0.5)
        return 0

    s._weg2_release_dormant_hold = slow
    s._weg2_hold_release_due = True
    with mock.patch.dict(os.environ, {"SGLANG_WEG2_RPC_STALL_WATCHDOG_S": "0.2",
                                      "SGLANG_WEG2_EVIDENCE_DIR": str(tmp_path),
                                      "SGLANG_WEG2_GROUP": "D"}):
        s._weg2_run_deferred_hold_release()
    files = list(tmp_path.glob("weg2_rpcstall_D_r0_hold_release_*.txt"))
    assert len(files) == 1 and "Thread" in files[0].read_text()


def test_the_resume_marks_the_release_due_under_the_switch():
    src = inspect.getsource(WU)
    i = src.index("SGLANG_WEG2_HOLD_RELEASE_AFTER_REPLY.get()")
    block = src[i:i + 1400]
    assert "scheduler._weg2_hold_release_due = True" in block
    # the in-RPC call is the switch-off arm only
    j = block.index("elif callable(_rel):")
    assert "_rel()" in block[j:] and "_rel()" not in block[:j]
    from sglang.srt.environ import envs
    assert envs.SGLANG_WEG2_HOLD_RELEASE_AFTER_REPLY.get() is True


def test_the_call_site_follows_the_reply_send():
    src = inspect.getsource(S.Scheduler.process_input_requests)
    send = src.index("send_to_tokenizer.send_output(output, recv_req)")
    run = src.index("self._weg2_run_deferred_hold_release()")
    loop = src.index("for recv_req in recv_reqs:")
    after_loop = src.index("self.flush_wrapper.check_pending()")
    assert loop < send < run < after_loop


# ---------------------------------------------------------------- per-rid hold read time

def test_segment_ms():
    t0 = 100.0
    marks = {"lockref": 100.001, "bind": 100.002, "alloc": 100.012, "vote0": 100.5, "vote1": 115.5}
    seg = PL.segment_ms(marks, t0, 115.6)
    assert seg == {"pre": 1.0, "bind": 1.0, "alloc": 10.0, "query": 488.0, "collective": 15000.0,
                   "post": 100.0, "total": 15600.0}
    seg = PL.segment_ms({}, t0, 100.004)                  # an ineligible read: no registration
    assert seg["pre"] is None and seg["collective"] is None and seg["total"] == 4.0


def test_issue_deferred_reads_logs_one_line_per_rid(caplog):
    from sglang.srt.mem_cache import unified_radix_cache as U

    cache = types.SimpleNamespace()

    def prefetch(req):
        for m in ("lockref", "bind", "alloc", "vote0", "vote1"):
            U._weg2_pfs_mark(cache, m)
        return "issued"

    sched = types.SimpleNamespace(tree_cache=cache, _prefetch_kvcache=prefetch)
    hold = []
    for i in range(2):
        r = types.SimpleNamespace(rid="weg2-24-7%d" % i)
        setattr(r, PL.DEFER_ATTR, True)
        hold.append(r)
    with caplog.at_level(logging.INFO, logger=PL.logger.name):
        out = PL.issue_deferred_reads(sched, hold)
    lines = [r.getMessage() for r in caplog.records if "WEG2-HOLD-READ-TIME" in r.getMessage()]
    assert len(out) == 2 and len(lines) == 2
    assert "rid=weg2-24-70" in lines[0] and "collective_ms=" in lines[0] and "None" not in lines[0]
    assert "_weg2_pfs_marks" not in cache.__dict__        # disarmed after each read
    U._weg2_pfs_mark(cache, "bind")                        # unarmed: a no-op
    assert "_weg2_pfs_marks" not in cache.__dict__


def test_the_marks_sit_in_prefetch_from_storage():
    from sglang.srt.mem_cache import unified_radix_cache as U

    src = inspect.getsource(U.UnifiedRadixCache.prefetch_from_storage)
    order = [src.index('_weg2_pfs_mark(self, "%s")' % m) for m in ("lockref", "bind", "alloc", "vote0", "vote1")]
    assert order == sorted(order)
    assert src.index('_weg2_pfs_mark(self, "vote0")') < src.index('label="prefetch_participation_vote"') \
        < src.index('_weg2_pfs_mark(self, "vote1")')
