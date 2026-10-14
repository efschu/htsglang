# SPDX-License-Identifier: Apache-2.0
"""PRIORITY LANES, fix round 5 (NF metal 6, image int23lanes5 @ f444d6dd9a, boot 10:38:26Z, done/lanes-metall6-1012.md).

The finding (BLOCKER, probe 2b variant (a): lane 0 = 98 000 tokens chunked on P, lane 1 = 16 000): fix 4 parks lane 0 correctly
(``PDFLIP-PARK (lane) span=65536 fwd_ct=5`` on PP0/PP1/PP2), the front takes it away (``LANE-P-TAKE`` -> ``#801 ABORT RECEIVED`` on all three
stages), P sleeps, the floor falls, lane 0 comes back under the SAME rid as a NEW request object -- and PP1/PP2 die with
``PdFlipStoreToldMismatch: #1400 STORE-TOLD MISMATCH ... told=65536 own_prefix=32768``.

The first reading of the run (write-back of the parked rows on the followers racing the P sleep) is REFUTED by the logs: the store held all
1024 parked pages when the stages came back (``#1439 ARENA-PRESENT n=4 keys=1531 leading_complete=1024`` on PP0, ``complete=1319 claimed=0``)
and the followers had asked for 512 only (``HANDOFF-KEYS REG ... new=32769 span=513`` on PP1/PP2 against ``new=97999`` on PP0).  Why: the
FIRST instance of the rid (before the abort) had put a paced read-ahead ``told=32768`` on the wire (PP0's read at the park saw 512 of 1024
pages: ``#1028B FETCH CAP ... claimed=512``); the abort came before the Admit (``#1416e PACED-DROP``), so the followers never consumed
``_pdflip_told_early[rid] = 32768``; at the resume ``pdflip_store_told.intake`` found that STALE entry for the new instance and registered a read
limited to it.  Variant (b) carries the same mechanism (``new=49153`` on PP1/PP2, the Admit entry ``_pdflip_store_told[rid] = 49152``) and is green
only because the second read of PP0 equals the stale value.

What is pinned here: (1) the metal-6 sequence on three PP stages through the REAL ``pdflip_store_told`` functions (intake, paced publish,
follower absorb, admission) and the REAL ``Scheduler._abort_request_now``: abort inside the pacing window, resume under the same rid with the
store complete -- every stage admits at the same told (RED on f444d6dd9a with the #1400 text); (2) the same with the Admit entry and a second read
that differs; (3) variant (b) (equal values) stays green; (4) the abort/absorb order inside one pass; (5) scope: other rids keep their entries,
``abort_all`` drops all, PP0's held/pacing are left to ``pp0_publish``, a follower's held request object is dropped; (6) R1: without the lane
switch nothing changes; (7) the ``TOLD-FORGET`` marker; (8) the call site in ``_abort_request_now``.
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

from flliper.srt.managers import scheduler as sch  # noqa: E402
from flliper.srt.managers import pdflip_store_told as m  # noqa: E402
from flliper.srt.managers.io_struct import AbortReq  # noqa: E402

DT = 0.05            # one pass of wall time
STORE_SHORT = 32768  # what the store holds while the followers' write-back is still being claimed (metal 6: 512 pages)
STORE_FULL = 65536   # what it holds a few seconds later (1024 pages)
T_FULL = 1.5         # the write-back is complete from here on
RID = "pdflip-0-5"


class _LanesOn:
    """FLLIPER_PDFLIP_LANES=1 on group P with the park on (what a lane boot has)."""

    def __enter__(self):
        self.keys = ("FLLIPER_PDFLIP_LANES", "FLLIPER_PDFLIP_GROUP", "FLLIPER_PDFLIP_PARK")
        self.old = {k: os.environ.get(k) for k in self.keys}
        os.environ["FLLIPER_PDFLIP_LANES"] = "1"
        os.environ["FLLIPER_PDFLIP_GROUP"] = "P"
        os.environ.pop("FLLIPER_PDFLIP_PARK", None)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _TimedTree:
    """Tree double: a store read terminates at a wall-clock time; ``release_aborted_request`` drops its records."""

    def __init__(self, clock):
        self.clock = clock
        self.done_at = {}
        self.completed = {}
        self.loaded = {}
        self._pending_completed = {}
        self.is_eagle = False

    def start_read(self, rid, tokens, duration):
        self.done_at[rid] = self.clock() + duration
        self._pending_completed[rid] = int(tokens)

    def check_prefetch_progress(self, rid):
        if rid not in self.done_at:
            return True
        if self.clock() < self.done_at[rid]:
            return False
        del self.done_at[rid]
        tokens = self._pending_completed.pop(rid)
        self.completed[rid] = tokens
        self.loaded[rid] = tokens
        return True

    def completed_prefetch_tokens(self, rid):
        return self.completed.get(rid)

    @property
    def prefetch_loaded_tokens_by_reqid(self):
        return self.loaded

    def pop_prefetch_loaded_tokens(self, rid):
        return self.loaded.pop(rid, 0)

    def release_aborted_request(self, rid):
        for d in (self.done_at, self._pending_completed, self.completed, self.loaded):
            d.pop(rid, None)


class _Stage:
    """One PP stage of group P: the told-form fields of the scheduler plus what ``_abort_request_now`` touches."""

    def __init__(self, pp_rank, clock, store_hit, read_s):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.pp_flip_counters = None
        self.tree_cache = _TimedTree(clock)
        self.waiting_queue = []
        self.store_hit = store_hit  # () -> tokens the store holds NOW
        self.read_s = read_s
        self.admitted = []          # (planned PP0 pass, rid, prefix cap, credit)
        self.reads = []             # (rid, limit_tokens, tokens read)
        self.clock = clock
        # what _abort_request_now reads
        self._pdflip_intake_watch = None
        self.chunked_req = None
        self.anchor_tails = ()
        self.ipc_channels = MagicMock()
        self.disaggregation_mode = None
        self.grammar_manager = MagicMock()
        self.running_mbs = []
        self.mbs = []
        self.kv_session_offload = None

    def _pdflip_abort_dormant_hold(self, recv_req):
        pass

    def _pdflip_d_park_abort(self, recv_req):
        pass

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        hit = int(self.store_hit())
        if limit_tokens is not None:
            hit = min(hit, int(limit_tokens))
        self.reads.append((req.rid, limit_tokens, hit))
        if hit <= 0:
            return "declined:store_absent"
        self.tree_cache.start_read(req.rid, hit, self.read_s)
        return "issued"


def _req(rid):
    return SimpleNamespace(rid=rid, prefetch_deferred=None, mamba_pool_idx=None)


class _Ring:
    """Three stages, the request wire with its per-hop lag (stage r plans PP0's pass k at its own pass k + r), a pass driver."""

    def __init__(self, monkeypatch, store_hit=None, metal_form=False):
        """``metal_form``: the form of the metal boot -- absolute told and the followers' early read at intake
        (``FOLLOWER-EARLY-READ``, ``early:own_read``); off: the span-relative told, a follower registers when the told arrives."""
        self.clock = _Clock()
        monkeypatch.setattr(m, "_clock", self.clock)
        monkeypatch.setenv(m.ENV_PACED, "1")
        monkeypatch.setenv("FLLIPER_PDFLIP_TOLD_ABSOLUTE", "1" if metal_form else "0")
        monkeypatch.setenv("FLLIPER_PDFLIP_FOLLOWER_EARLY_READ", "1" if metal_form else "0")
        # the pipeline is busy in the metal run (lane 1's chunk is in flight): the window is kept, no PACED-IDLE-ADMIT
        monkeypatch.setenv("FLLIPER_PDFLIP_DISABLE_TOLD_PACE_IDLE_SKIP", "1")
        monkeypatch.setattr(m.time, "sleep", lambda sec: setattr(self.clock, "t", self.clock.t + sec))
        self.store_hit = store_hit or (lambda: STORE_SHORT if self.clock.t < T_FULL else STORE_FULL)
        self.stages = [_Stage(r, self.clock, lambda: self.store_hit(), 0.2 + 0.1 * r) for r in range(3)]
        for s in self.stages:
            assert m.armed(s)
        self.wire = {}
        self.pass_n = 0
        self.aborts = []  # (PP0 pass at which the abort is dispatched, AbortReq)
        self.errors = []

    def arrive(self, rid):
        for s in self.stages:
            r = _req(rid)
            s.waiting_queue.append(r)
            m.intake(s, r, lambda gate: None)

    def abort(self, rid, abort_all=False):
        """The front's /abort_request: PP0 dispatches it in its next pass, stage r in the pass that plans PP0's (r passes later)."""
        self.aborts.append((self.pass_n, AbortReq(rid=rid, abort_all=abort_all)))

    def step(self):
        k = self.pass_n
        self.wire[k] = m.pp0_publish(self.stages[0], [])
        for s in self.stages[1:]:
            src = k - s.ps.pp_rank
            if src >= 0:
                m.follower_absorb(s, list(self.wire.get(src, [])))
        for s in self.stages:
            for k0, recv in self.aborts:
                if k - s.ps.pp_rank == k0:
                    sch.Scheduler._abort_request_now(s, recv)
        for s in self.stages:
            for req in list(s.waiting_queue):
                try:
                    credit = m.admission(s, req, lambda kind, rid: None)
                except m.PdFlipStoreToldMismatch as exc:
                    self.errors.append((s.ps.pp_rank, str(exc)))
                    s.waiting_queue.remove(req)
                    continue
                if credit is None:
                    continue
                s.waiting_queue.remove(req)
                s.admitted.append((k - s.ps.pp_rank, req.rid, getattr(req, "_pdflip_prefix_cap", None), credit))
        self.pass_n += 1
        self.clock.t += DT

    def run(self, passes):
        for _ in range(passes):
            self.step()

    def run_until(self, cond, limit=400):
        for _ in range(limit):
            if cond():
                return
            self.step()
        raise AssertionError("condition not reached in %d passes" % limit)

    def plans(self, rid):
        return [[a[:3] for a in s.admitted if a[1] == rid] for s in self.stages]


# ---------------------------------------------------------------------------
# 1. metal 6 variant (a): the paced read-ahead of the first instance outlives its abort
# ---------------------------------------------------------------------------

def _abort_inside_the_pacing_window(ring):
    """Instance 1 of RID: the read-ahead told (the store holds STORE_SHORT) is on every follower; the front aborts before the Admit."""
    ring.arrive(RID)
    ring.run_until(lambda: all(RID in getattr(s, "_pdflip_told_early", {}) for s in ring.stages[1:]))
    assert ring.stages[1]._pdflip_told_early[RID] == STORE_SHORT          # the follower saw PP0's short read-ahead (metal: told=32768)
    assert not any(a[1] == RID for s in ring.stages for a in s.admitted)  # ... and no Admit yet (PACED-DROP follows on PP0)
    ring.abort(RID)
    ring.run(4)  # every stage dispatched the abort; PP0 retired its pacing entry (PACED-DROP)
    assert all(not s.waiting_queue for s in ring.stages)


@pytest.mark.parametrize("metal_form", [False, True], ids=["relative-told", "metal-form-absolute-early-read"])
def test_resume_under_the_same_rid_does_not_inherit_the_first_told(monkeypatch, metal_form):
    """RED on f444d6dd9a: ``PdFlipStoreToldMismatch: #1400 STORE-TOLD MISMATCH rank pp=1/2 ... told=65536 own_prefix=32768`` -- the follower
    registered its read of the new instance at the first instance's told (``HANDOFF-KEYS REG ... new=32769``)."""
    with _LanesOn():
        ring = _Ring(monkeypatch, metal_form=metal_form)
        _abort_inside_the_pacing_window(ring)
        ring.run_until(lambda: ring.clock.t > T_FULL)     # the write-back of the parked span is complete (metal: leading_complete=1024)
        ring.arrive(RID)                                   # the lane resume: SAME rid, NEW request objects
        ring.run_until(lambda: ring.errors or all(len(p) == 1 for p in ring.plans(RID)))
        assert not ring.errors, ring.errors
        plans = ring.plans(RID)
        assert plans[0] == plans[1] == plans[2], plans     # same PP0 pass, same prefix cap on every stage
        assert plans[0][0][2] == STORE_FULL, plans         # the whole parked span, not the first instance's 32768
        assert [a[3] for s in ring.stages for a in s.admitted if a[1] == RID] == [STORE_FULL] * 3
        # the followers' read of instance 2 is bounded by the NEW told only (65536), never by the first one (32768)
        for s in ring.stages[1:]:
            limits = [lim for rid, lim, _ in s.reads if rid == RID]
            assert STORE_SHORT not in limits[1:], (s.ps.pp_rank, s.reads)


def test_the_stale_entry_is_the_read_ahead_not_a_store_state(monkeypatch):
    """The refuted reading, pinned: with the ledger stale the follower's second read stops at the FIRST told although the store is complete;
    this is what separates 'write-back race' (the store would be short) from 'stale told' (the store is complete, the read is capped)."""
    ring = _Ring(monkeypatch)
    ring.arrive(RID)
    ring.run_until(lambda: all(RID in getattr(s, "_pdflip_told_early", {}) for s in ring.stages[1:]))
    ring.abort(RID)
    ring.run(4)
    ring.run_until(lambda: ring.clock.t > T_FULL)
    assert ring.store_hit() == STORE_FULL                            # the store is complete ...
    # (no lanes switch here: the base behaviour -- the entry is still in the ledger)
    assert ring.stages[1]._pdflip_told_early.get(RID) == STORE_SHORT   # ... and the follower still carries the first instance's told
    ring.arrive(RID)
    assert ring.stages[1].reads[-1][:2] == (RID, STORE_SHORT)        # the read of instance 2 is limited to it: 'new=32769' of the log


# ---------------------------------------------------------------------------
# 2. the Admit entry (variant b's mechanism) and the case where it is harmless
# ---------------------------------------------------------------------------

def test_admit_entry_with_a_different_second_read_is_not_inherited(monkeypatch):
    """The Admit entry of instance 1 (told = the short read) against a second read that is longer: RED on f444d6dd9a (mismatch)."""
    with _LanesOn():
        ring = _Ring(monkeypatch, store_hit=lambda: STORE_SHORT if ring.clock.t < T_FULL else STORE_FULL)
        ring.arrive(RID)
        # stop the harness's own admission of instance 1: PP0 admits at the Admit pass, which is before the abort can reach it, so
        # emulate the lane hold (the floor keeps the request in the queue) by driving the passes by hand up to the Admit absorb
        for _ in range(400):
            ring.wire[ring.pass_n] = m.pp0_publish(ring.stages[0], [])
            k = ring.pass_n
            for s in ring.stages[1:]:
                if k - s.ps.pp_rank >= 0:
                    m.follower_absorb(s, list(ring.wire.get(k - s.ps.pp_rank, [])))
            ring.pass_n += 1
            ring.clock.t += DT
            if all(RID in s._pdflip_store_told for s in ring.stages):
                break
        assert all(s._pdflip_store_told.get(RID) == STORE_SHORT for s in ring.stages)
        for s in ring.stages:  # the front takes the lane-held request away: the abort reaches stage r r passes after PP0
            sch.Scheduler._abort_request_now(s, AbortReq(rid=RID))
        assert all(not s.waiting_queue for s in ring.stages)
        ring.run_until(lambda: ring.clock.t > T_FULL)
        ring.arrive(RID)
        ring.run_until(lambda: ring.errors or all(len(p) == 1 for p in ring.plans(RID)))
        assert not ring.errors, ring.errors
        plans = ring.plans(RID)
        assert plans[0] == plans[1] == plans[2], plans
        assert plans[0][0][2] == STORE_FULL


def test_variant_b_equal_values_stay_green(monkeypatch):
    """Metal 6 boot 2 (variant b, must stay green): the first instance's Admit entry equals what PP0 reads the second time (49152 = 49152).
    After the fix every stage admits the resumed request at the same PP0 pass with the same cap.  (On f444d6dd9a this shape does not die in
    the harness, but the stages are not in step either: PP0 admits on the stale entry one pass apart from the followers -- the same stale
    ledger, a milder face; the metal run (b) was green, why the skew did not bite there was not examined: unverified.)"""
    with _LanesOn():
        ring = _Ring(monkeypatch, store_hit=lambda: STORE_SHORT)  # the store never grows: both reads see the same count
        ring.arrive(RID)
        for _ in range(400):
            k = ring.pass_n
            ring.wire[k] = m.pp0_publish(ring.stages[0], [])
            for s in ring.stages[1:]:
                if k - s.ps.pp_rank >= 0:
                    m.follower_absorb(s, list(ring.wire.get(k - s.ps.pp_rank, [])))
            ring.pass_n += 1
            ring.clock.t += DT
            if all(RID in s._pdflip_store_told for s in ring.stages):
                break
        for s in ring.stages:
            sch.Scheduler._abort_request_now(s, AbortReq(rid=RID))
        ring.arrive(RID)
        ring.run_until(lambda: ring.errors or all(len(p) == 1 for p in ring.plans(RID)))
        assert not ring.errors, ring.errors
        plans = ring.plans(RID)
        assert plans[0] == plans[1] == plans[2] and plans[0][0][2] == STORE_SHORT


# ---------------------------------------------------------------------------
# 3. order inside one pass, scope, held / pacing
# ---------------------------------------------------------------------------

def test_a_told_absorbed_in_the_pass_of_the_abort_is_dropped_by_it(monkeypatch):
    """PP0 publishes at the top of the pass, BEFORE it dispatches the abort: the follower's list is [AbortReq, ..., PdFlipStoreTold] and its
    absorb runs before the dispatch.  The entry the absorb makes is the one the abort drops."""
    with _LanesOn():
        ring = _Ring(monkeypatch)
        ring.arrive(RID)
        pp1 = ring.stages[1]
        told = m.PdFlipStoreTold(rid=RID, told=STORE_SHORT, paced=True)
        m.follower_absorb(pp1, [AbortReq(rid=RID), told])        # the absorb takes the told off the list ...
        assert pp1._pdflip_told_early[RID] == STORE_SHORT
        sch.Scheduler._abort_request_now(pp1, AbortReq(rid=RID))  # ... and the dispatch of the same pass drops it
        assert RID not in pp1._pdflip_told_early
        assert not pp1.waiting_queue


def test_other_rids_keep_their_entries_and_abort_all_drops_all(monkeypatch):
    with _LanesOn():
        ring = _Ring(monkeypatch)
        pp1 = ring.stages[1]
        m._early(pp1).update({RID: 1, "pdflip-1-6": 2})
        pp1._pdflip_store_told.update({RID: 3, "pdflip-1-6": 4})
        sch.Scheduler._abort_request_now(pp1, AbortReq(rid=RID))
        assert pp1._pdflip_told_early == {"pdflip-1-6": 2} and pp1._pdflip_store_told == {"pdflip-1-6": 4}
        sch.Scheduler._abort_request_now(pp1, AbortReq(rid="", abort_all=True))
        assert pp1._pdflip_told_early == {} and pp1._pdflip_store_told == {}


def test_pp0_held_and_pacing_are_left_to_the_publish_and_the_followers_held_request_is_dropped(monkeypatch):
    with _LanesOn():
        ring = _Ring(monkeypatch)
        ring.arrive(RID)
        pp0, pp1 = ring.stages[0], ring.stages[1]
        assert RID in pp0._pdflip_store_held and RID in pp1._pdflip_store_held
        for s in (pp0, pp1):
            sch.Scheduler._abort_request_now(s, AbortReq(rid=RID))
        assert RID in pp0._pdflip_store_held          # PP0: retired by pp0_publish (PACED-DROP / twin / PF bookkeeping), not here
        assert RID not in pp1._pdflip_store_held      # the follower's held entry named the request object the abort deleted
        # a told that arrives for the dropped request must not register a read for it
        reads_before = len(pp1.reads)
        m.follower_absorb(pp1, [m.PdFlipStoreTold(rid=RID, told=STORE_SHORT, paced=True)])
        assert len(pp1.reads) == reads_before


def test_nothing_is_dropped_when_the_told_form_is_not_armed():
    sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=1), _pdflip_told_early={RID: 5}, _pdflip_store_told={RID: 5})
    assert m.forget_aborted(sched, RID) == {}
    assert sched._pdflip_told_early == {RID: 5}


# ---------------------------------------------------------------------------
# 4. R1: the default path
# ---------------------------------------------------------------------------

def test_r1_without_the_lane_switch_the_ledger_is_left_exactly_as_before(monkeypatch):
    """No FLLIPER_PDFLIP_LANES: ``forget_aborted_told`` returns at once and the abort leaves the ledger alone (the latent stale entry of the
    default path is NOT changed by this fix; reported, not touched)."""
    from flliper.srt.pdflip import lanes_p

    monkeypatch.delenv("FLLIPER_PDFLIP_LANES", raising=False)
    ring = _Ring(monkeypatch)
    ring.arrive(RID)
    ring.run_until(lambda: all(RID in getattr(s, "_pdflip_told_early", {}) for s in ring.stages[1:]))
    pp1 = ring.stages[1]
    before = dict(pp1._pdflip_told_early)
    assert lanes_p.forget_aborted_told(pp1, AbortReq(rid=RID)) is None
    sch.Scheduler._abort_request_now(pp1, AbortReq(rid=RID))
    assert pp1._pdflip_told_early == before


def test_r1_group_d_is_not_touched(monkeypatch):
    from flliper.srt.pdflip import lanes_p

    with _LanesOn():
        os.environ["FLLIPER_PDFLIP_GROUP"] = "D"
        sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=1), _pdflip_store_told_armed=True, _pdflip_told_early={RID: 5})
        assert lanes_p.forget_aborted_told(sched, AbortReq(rid=RID)) is None
        assert sched._pdflip_told_early == {RID: 5}


# ---------------------------------------------------------------------------
# 5. the marker, the call site
# ---------------------------------------------------------------------------

def test_the_forget_is_named_in_the_log(monkeypatch, caplog):
    from flliper.srt.pdflip import lanes_p

    with _LanesOn():
        ring = _Ring(monkeypatch)
        ring.arrive(RID)
        ring.run_until(lambda: all(RID in getattr(s, "_pdflip_told_early", {}) for s in ring.stages[1:]))
        with caplog.at_level(logging.INFO):
            sch.Scheduler._abort_request_now(ring.stages[2], AbortReq(rid=RID))
        lines = [r.getMessage() for r in caplog.records if lanes_p.MARK_TOLD_FORGET in r.getMessage()]
        assert len(lines) == 1, lines
        assert RID in lines[0] and "'early': %d" % STORE_SHORT in lines[0] and "pp=2" in lines[0], lines[0]
        # nothing to drop -> no line
        caplog.clear()
        with caplog.at_level(logging.INFO):
            sch.Scheduler._abort_request_now(ring.stages[2], AbortReq(rid=RID))
        assert not [r for r in caplog.records if lanes_p.MARK_TOLD_FORGET in r.getMessage()]


def test_a_failing_forget_never_blocks_the_abort(monkeypatch, caplog):
    from flliper.srt.pdflip import lanes_p

    with _LanesOn():
        ring = _Ring(monkeypatch)
        ring.arrive(RID)

        def boom(*a, **k):
            raise RuntimeError("ledger broke")

        monkeypatch.setattr(m, "forget_aborted", boom)
        with caplog.at_level(logging.WARNING):
            sch.Scheduler._abort_request_now(ring.stages[1], AbortReq(rid=RID))
        assert not ring.stages[1].waiting_queue                      # the abort itself went through
        assert any("FAILED" in r.getMessage() and lanes_p.MARK_TOLD_FORGET in r.getMessage() for r in caplog.records)


def test_the_call_sits_in_abort_request_now_after_the_waiting_queue_deletion():
    src = textwrap.dedent(inspect.getsource(sch.Scheduler._abort_request_now))
    assert src.count("_lanes_p.forget_aborted_told(self, recv_req)") == 1
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_lanes_p.forget_aborted_told"]
    assert len(calls) == 1
    # after the loop that pops the aborted requests from the waiting queue, before the running-batch abort
    assert src.index("self.waiting_queue.pop(i)") < src.index("_lanes_p.forget_aborted_told") < src.index("Delete requests in the running batch")
