"""fnFL2 H42c: the P burst-assembly hold decides on ONE clock (PP0's).

THE DEATH (x161, tree 7370b10062, 24.09. 15:41:43). Carrierless P form
(``#631 ROW AUTHORITY DISABLED``): every stage plans its own admission in the
logical pass whose request list it just received, so every stage also took
H42b's burst verdict -- each on its own ``time.monotonic()``. After the 97k
request (7 forwards: 6 chunks + a 1-token END-ANCHOR tail, slots agreeing on
all stages) the 12672-token rid weg2-2-6 arrived; PP1 released its hold one
logical pass before PP0 (``held_ms=53`` vs ``55``) and admitted the body
[0, 12668) in slot 2 while PP0 ran it in slot 0. PP0's proxy stamped slot 0
reached PP1's slot 2 (``#1004 IDENTITY REFUSAL BYPASSED ... stamp mb_id=0 ...
while on mb_id=2``), the bypass returned None and the model died on a
nameless "no pp_proxy_tensors".

WHAT IS PINNED.
* The wire: PP0 stamps its pass clock onto the list it SENDS (the dispatched
  list stays clean), a follower relays it verbatim and takes it off before
  dispatch -- through the real ``_pp_forward_and_process_input_requests``.
* The sequence of x161 on a simulated PP0 and PP1 at width 4 with a follower
  whose wall time drifts behind PP0's: the body and the tail of weg2-2-6 land
  in the SAME slot with the SAME fwd_ct on both stages. The mutant (each stage
  reads its own clock -- the shipped H42b behaviour) reproduces x161's split,
  so the pin can fail.
* A PP stage without PP0's clock refuses by name instead of deciding.
* #1004: a slot mismatch on a one-layout process is a named stop, not None.
"""

import inspect
import pickle
import time as _time
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.managers import anchor_tails as at
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

PP = 3
PP_LOOP = 3  # the rig's slot ring (#969M ARM mb_id 0,1,2)
CHUNK = 16384
WINDOW_MS = 300
A_RID, A_TOK = "weg2-0-4", 97841
A_CHUNKS = [16384] * 5 + [15920, 1]  # 6 chunks + the 1-token END-ANCHOR tail
B_RID, B_TOK = "weg2-2-6", 12672
B_CHUNKS = [12668, 4]  # END-ANCHOR SPLIT n=2: anchor at 12668, grain 4
LAG_GROWTH_MS = 0.05  # PP1's drift behind PP0 per idle pass during the hold


def _scheduler_cls():
    from sglang.srt.managers.scheduler import Scheduler

    return Scheduler


def _mixin_cls():
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    return SchedulerPPMixin


# ------------------------------------------------------------ helpers
def test_clock_rides_the_wire_and_leaves_it():
    wire = at.stamp_burst_clock(["req"], 12.5)
    assert wire[0] == "req" and isinstance(wire[1], at.Weg2BurstClock)
    back = pickle.loads(pickle.dumps(wire))  # the chain pickles the list whole
    rest, now = at.absorb_burst_clock(back)
    assert rest == ["req"] and now == 12.5
    assert at.absorb_burst_clock(["req"]) == (["req"], None)
    assert at.without_burst_clock(back) == ["req"]
    with pytest.raises(at.BurstClockAbsent, match="2 PP0 pass clocks"):
        at.absorb_burst_clock(back + [at.Weg2BurstClock(13.0)])


def test_pass_now_is_pp0s_on_a_pp_group_and_local_otherwise():
    assert at.burst_pass_now(1, None, lambda: 7.0) == 7.0
    assert at.burst_pass_now(3, 4.25, lambda: 7.0) == 4.25
    with pytest.raises(at.BurstClockAbsent, match="BURST-CLOCK ABSENT"):
        at.burst_pass_now(3, None, lambda: 7.0)


def test_clock_is_armed_only_with_the_window_on_a_pp_group():
    with envs.SGLANG_WEG2_P_BURST_ASSEMBLY_MS.override(WINDOW_MS):
        assert at.burst_clock_armed(3) and not at.burst_clock_armed(1)
    with envs.SGLANG_WEG2_P_BURST_ASSEMBLY_MS.override(0):
        assert not at.burst_clock_armed(3)


# ------------------------------------------------------------ the wire hop
class _Stage:
    """One P stage around the real request-wire hop and the real hold."""

    def __init__(self, pp_rank):
        self.pp_rank = pp_rank
        self.sent = []  # lists this stage put on the chain
        self.dispatched = []  # lists this stage handed to process_input_requests
        h = SimpleNamespace()
        h.ps = SimpleNamespace(pp_size=PP, pp_rank=pp_rank, attn_tp_rank=0, attn_cp_rank=0)
        h.pp_group = SimpleNamespace(is_first_rank=pp_rank == 0, is_last_rank=pp_rank == PP - 1)
        h.send_req_work = None
        h._weg2_store_told_armed = False
        h._weg2_vote_pass_hook = lambda reqs: None
        h._weg2_vote_after_forward = lambda reqs: reqs
        h._pp_commit_comm_work = lambda work: None
        h._pp_send_pyobj_to_next_stage = lambda reqs, async_send=True: self.sent.append(
            pickle.loads(pickle.dumps(list(reqs)))
        )
        h.process_input_requests = self._process
        h.waiting_queue = []
        h.get_num_allocatable_reqs = lambda running_bs: 4  # --max-running-requests 4
        self.h = h
        self.fwd_ct = 0
        self.carried = None  # (req, remaining chunks, next start)
        self.log = []  # (rid, start, end, slot, fwd_ct)
        self.holds = 0

    def _process(self, reqs):
        self.dispatched.append(list(reqs))
        self.h.waiting_queue += [r for r in reqs if isinstance(r, SimpleNamespace)]

    def intake(self, reqs):
        _mixin_cls()._pp_forward_and_process_input_requests(self.h, reqs)

    def plan(self, m):
        """get_next_batch_to_run of logical pass m: a carried chunk runs with
        no verdict (reason 'carried'); otherwise the H42b gate decides."""
        slot = m % PP_LOOP
        if self.carried is None and self.h.waiting_queue:
            adder = SimpleNamespace(can_run_list=[], rem_chunk_tokens=CHUNK)
            hold = _scheduler_cls()._weg2_burst_assembly_hold(
                self.h, adder, SimpleNamespace(reqs=[])
            )
            if hold is not None:
                self.holds += 1
                return
            req = self.h.waiting_queue.pop(0)
            chunks = list(A_CHUNKS if req.rid == A_RID else B_CHUNKS)
            self.carried = (req, chunks, 0)
        if self.carried is None:
            return
        req, chunks, start = self.carried
        n = chunks.pop(0)
        self.log.append((req.rid, start, start + n, slot, self.fwd_ct))
        self.fwd_ct += 1
        self.carried = (req, chunks, start + n) if chunks else None


def _run(monkeypatch, *, local_clock=False):
    """x161's order on PP0 + PP1. PP0 runs ahead of PP1 on the async chain
    send; PP1's lag grows while both idle through the hold (PP1 1 ms behind at
    the arrival, +0.05 ms per 2 ms pass) -- the drift that put PP1's quiet
    verdict one pass ahead of PP0's in x161 (held_ms 53 vs 55)."""
    wall = [0.0]
    monkeypatch.setattr(_time, "monotonic", lambda: wall[0])
    if local_clock:  # MUTANT: every stage reads its own clock (shipped H42b)
        monkeypatch.setattr(at, "burst_pass_now", lambda pp, clk, mono: float(mono()))
    stages = [_Stage(0), _Stage(1)]
    arrive = {2: SimpleNamespace(rid=A_RID, origin_input_ids=[0] * A_TOK),
              20: SimpleNamespace(rid=B_RID, origin_input_ids=[0] * B_TOK)}
    t0 = 1000.0
    b_seen = None
    with envs.SGLANG_WEG2_P_BURST_ASSEMBLY_MS.override(WINDOW_MS):
        for m in range(140):
            new =[arrive[m]] if m in arrive else []
            if m == 20:
                b_seen = m
            lag_ms = 3.0 if b_seen is None else 1.0 + LAG_GROWTH_MS * (m - b_seen)
            n_fwd = len(stages[0].log)
            # PP0: tokenizer list -> stamp + send -> dispatch -> plan
            wall[0] = t0
            stages[0].intake(list(new))
            stages[0].plan(m)
            # PP1: PP0's list m (as it crossed the wire) -> relay -> plan
            wall[0] = t0 + lag_ms / 1000.0
            stages[1].intake(stages[0].sent[-1])
            stages[1].plan(m)
            t0 += 4.0 if len(stages[0].log) > n_fwd else 0.002  # forward vs idle pass
    return stages


def _slots(stage, rid):
    return [(s, e, slot, fc) for r, s, e, slot, fc in stage.log if r == rid]


def test_the_wire_hop_relays_the_clock_and_dispatches_without_it(monkeypatch):
    stages = _run(monkeypatch)
    pp0, pp1 = stages
    assert all(isinstance(lst[-1], at.Weg2BurstClock) for lst in pp0.sent)
    assert not any(isinstance(r, at.Weg2BurstClock) for lst in pp0.dispatched for r in lst)
    # PP1 relays list m verbatim to PP2, clock included, and dispatches it clean
    assert [len(x) for x in pp1.sent] == [len(x) for x in pp0.sent]
    assert all(isinstance(lst[-1], at.Weg2BurstClock) for lst in pp1.sent)
    assert not any(isinstance(r, at.Weg2BurstClock) for lst in pp1.dispatched for r in lst)
    assert pp1.h._weg2_burst_clock == pp0.h._weg2_burst_clock


def test_x161_sequence_keeps_one_slot_per_forward_on_every_stage(monkeypatch):
    pp0, pp1 = _run(monkeypatch)
    a0, a1 = _slots(pp0, A_RID), _slots(pp1, A_RID)
    assert len(a0) == 7 and a0 == a1  # 97k: 7 forwards, same slots, as in x161
    b0, b1 = _slots(pp0, B_RID), _slots(pp1, B_RID)
    assert [(s, e) for s, e, _, _ in b0] == [(0, 12668), (12668, 12672)]
    assert pp0.holds > 0  # the hold was really exercised (12672 < 16384)
    assert b0 == b1, (b0, b1)  # body AND tail: same slot, same fwd_ct
    assert b0[0][3] == 7  # the x161 fwd_ct of the body


def test_mutant_local_clock_reproduces_the_x161_split(monkeypatch):
    """The can-fail proof: with each stage's own clock the follower releases
    one pass early and admits the body in the slot BEFORE PP0's."""
    pp0, pp1 = _run(monkeypatch, local_clock=True)
    assert _slots(pp0, A_RID) == _slots(pp1, A_RID)  # the 97k run is unaffected
    b0, b1 = _slots(pp0, B_RID), _slots(pp1, B_RID)
    assert b0 != b1
    assert b1[0][2] == (b0[0][2] - 1) % PP_LOOP  # x161: PP1 slot 2, PP0 slot 0
    assert b1[0][3] == b0[0][3] == 7  # same fwd_ct, different slot -- x161's lines


def test_a_pp_stage_without_pp0s_clock_refuses(monkeypatch):
    stage = _Stage(1)
    stage.h.waiting_queue = [SimpleNamespace(rid=B_RID, origin_input_ids=[0] * B_TOK)]
    stage.h._weg2_burst_clock = None
    with envs.SGLANG_WEG2_P_BURST_ASSEMBLY_MS.override(WINDOW_MS):
        with pytest.raises(at.BurstClockAbsent):
            stage.plan(0)


# ------------------------------------------------------------ #1004
def test_1004_slot_mismatch_is_a_named_stop():
    from sglang.srt.managers import scheduler_pp_mixin as m

    src = inspect.getsource(m.SchedulerPPMixin._pp_recv_proxy_tensors)
    assert "IDENTITY REFUSAL BYPASSED" not in src
    tail = src[src.index("self._pp_proxy_drops = getattr"):]
    assert "raise RuntimeError(" in tail and "pp_slot_disagreement_message(" in tail
    assert "return None" not in tail.split("pp_slot_disagreement_message(")[0]

    batch = SimpleNamespace(reqs=[SimpleNamespace(rid=B_RID)], extend_num_tokens=12668)
    msg = m.pp_slot_disagreement_message(
        pp_rank=1, mb_id=2, stamp=(0, 8, 12668, -1, 8, (B_RID, 0, 12668)),
        recv_fwd_ct=7, batch=batch,
    )
    assert "#1004 SLOT DISAGREEMENT" in msg
    assert "PP1 is launching slot 2 (fwd_ct=7, rids=[weg2-2-6], extend=12668)" in msg
    assert "proxy names slot 0 (seq=8 rows=12668 sender_fwd_ct=8" in msg
    # a short / foreign stamp still yields a message, never a second exception
    assert "slot ?" in m.pp_slot_disagreement_message(
        pp_rank=2, mb_id=1, stamp=(), recv_fwd_ct=-1, batch=None
    )


def test_req_trace_ignores_the_clock():
    src = inspect.getsource(_mixin_cls()._pp_forward_and_process_input_requests)
    assert "_traced = _anchor_tails.without_burst_clock(recv_reqs)" in src
    stamp = src.index("_anchor_tails.stamp_burst_clock(")
    send = src.index("self._pp_send_pyobj_to_next_stage(")
    absorb = src.index("_anchor_tails.absorb_burst_clock(")
    dispatch = src.index("self.process_input_requests(recv_reqs)")
    assert stamp < send < absorb < dispatch
