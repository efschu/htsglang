"""P-PP3 overlap (20.10.2026): the younger set's told is acked BEFORE the followers compute the older row.

Metal (rc27 893d09fa3f, P log ..._1020_145331), queue waiting, PP0 idle between single forwards:
  16:37:21 PP0 '#TW TWIN-REGISTER rid=pdflip-34-634 twin=True verdict=issued head=91520 span=417'
  16:37:21 PP0 'S82 NO-PARK reason=no-waiter waiting=['pdflip-34-634', 'pdflip-34-635'] slot=(0, 1)'
  16:37:21 PP0 'PDFLIP P-HELD-REREAD rid=pdflip-34-634 n=1/2 have=0 verdict=issued why=hint hint=87040'
  16:37:24 PP0 '#1157 PREFETCH REAPED req=pdflip-34-634 ... hit_pages=0 completed=0 elapsed=3.03s ... policy=timeout'
  16:37:28 PP0 '#PGAP pp_rank=0 fwd=45 tokens=418 gpu_gap_ms=3133.3 ... output_commit=2876'
  16:37:27 PP0 'S82 COMMIT-PARK n=16 outcome=output rid=None waited_ms=743.6 slot=(0, 1) reads=1 acks=0'

Pinned (hermetic, CPU):
  1. P-HELD-REREAD compares the hint with the ABSOLUTE depth (head + span) -- the twin above is not re-read;
  2. the W2C pure verdicts (announce / owed / relay / key copy / named timeout);
  3. V1: the announced PP0 pass runs no S82 park and no MINIFWD hold before its chain send;
  4. the frame key: popped before compute, relayed by the middle rank (V2), absent without an announcement;
  5. THREE-RANK protocol run (real announce / defer / relay / pop code on three rank stand-ins, a step clock and two
     wires per hop): chain k absorbed before the older row on PP1 AND PP2, the younger set admitted on ``acks`` while
     the older row is still in flight, no deadlock; and the review mutants (park in pass k, no C3, no relay) go RED;
  6. the loop wiring in scheduler_pp_mixin (send, receive pop, probe defer, after-send check).
"""

import collections
import inspect
import os
import sys
import time as _time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_store_told as told  # noqa: E402
from flliper.srt.pdflip import pp_chain_announce as w2c  # noqa: E402
from flliper.srt.pdflip import short_read as sr  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# ---------------------------------------------------------------------------
# 1. P-HELD-REREAD: absolute depth


def _reread_case(head: int, have: int, hint: int, rid: str):
    calls = []
    tree = SimpleNamespace(completed_prefetch_tokens=lambda r: have, page_size=64)
    req = SimpleNamespace(rid=rid, _prefetch_registered_prefix_len=head, origin_input_ids=[1] * (hint + 512))
    sched = SimpleNamespace(page_size=64, pdflip_dormant=False,
                            _prefetch_kvcache=lambda r: (calls.append(r.rid), "issued")[1])
    sr.forget(rid)
    sr.note_hint(rid, hint)
    out = told._p_held_short_reread(sched, tree, req, rid)
    sr.forget(rid)
    return out, calls


def test_twin_whose_head_covers_the_hint_is_not_reread_metal_634():
    # '#TW TWIN-REGISTER rid=pdflip-34-634 ... head=91520' + 'P-HELD-REREAD ... have=0 ... hint=87040'
    out, calls = _reread_case(head=91520, have=0, hint=87040, rid="pdflip-34-634")
    assert out is False and calls == []


def test_intake_read_that_reached_the_hint_is_not_reread_metal_2_38():
    # 'STORE-TOLD INTAKE rid=pdflip-2-38 ... matched=18240', 'P-HELD-REREAD ... have=67712 ... hint=85952'
    out, calls = _reread_case(head=18240, have=67712, hint=85952, rid="pdflip-2-38")
    assert out is False and calls == []


def test_a_read_short_of_the_hint_is_still_reread():
    # the rule itself stays: head + span a page or more below D's measured depth -> re-read before the told
    out, calls = _reread_case(head=18240, have=67712, hint=99648, rid="pdflip-x-1")
    assert out is True and calls == ["pdflip-x-1"]


# ---------------------------------------------------------------------------
# 2. pure verdicts


def test_announce_only_with_a_publishable_read_and_nothing_else_in_flight():
    ok = dict(armed=True, others_in_flight=False, carried=False, publishable=["r"])
    assert w2c.pp0_should_announce(**ok) is True
    assert w2c.pp0_should_announce(**{**ok, "publishable": []}) is False
    assert w2c.pp0_should_announce(**{**ok, "others_in_flight": True}) is False
    assert w2c.pp0_should_announce(**{**ok, "carried": True}) is False
    assert w2c.pp0_should_announce(**{**ok, "armed": False}) is False
    assert w2c.announce_value(41) == 42


def test_owed_relay_and_key_copy():
    assert w2c.owed(None, 0) is False
    assert w2c.owed(42, 41) is True and w2c.owed(42, 42) is False
    assert w2c.absorbed(43, 1) == 42
    assert w2c.relay_value(None, 7) is None and w2c.relay_value(42, 7) == 7
    base = {"__admission_decision__": (0, ())}
    out = w2c.with_key(base, 5)
    assert out[w2c.KEY] == 5 and w2c.KEY not in base          # a copy: the relayed row keeps no stale key
    assert w2c.with_key(base, None) is base and w2c.with_key(None, None) is None


def test_deadline_is_a_named_789_stop():
    w2c.check_deadline(0.0, 1.0, 30.0, slot=1, k=3, absorbed_n=2)
    with pytest.raises(w2c.AnnouncedChainTimeout, match="#789 W2C ANNOUNCED-CHAIN TIMEOUT"):
        w2c.check_deadline(0.0, 31.0, 30.0, slot=1, k=3, absorbed_n=2)


# ---------------------------------------------------------------------------
# 3. V1 in the real pp0_publish


def test_announced_pass_runs_no_wait_before_its_chain(monkeypatch):
    seen = []
    monkeypatch.setattr(told._twin, "tick", lambda s: None)
    monkeypatch.setattr(told, "_minifwd_hold", lambda s, r: seen.append("minifwd"))
    monkeypatch.setattr(told, "_commit_park", lambda s, r: seen.append("park"))
    monkeypatch.setattr(told, "_pp0_publish_paced", lambda s, r: list(r))
    sched = SimpleNamespace(_pdflip_store_held={}, _pdflip_told_paced_on=True, _w2c_announced=(9, _time.monotonic()))
    told.pp0_publish(sched, [])
    assert seen == [] and sched._w2c_announced is None and sched._w2c_sending[0] == 9
    told.pp0_publish(sched, [])                                # the next pass is an ordinary pass again
    assert seen == ["minifwd", "park"]


# ---------------------------------------------------------------------------
# 4. frame key: pop, relay, absence


class _Ctr:
    def __init__(self, sent=0, consumed=0):
        self.s, self.c = sent, consumed

    def local_sent(self, chan):
        return self.s

    def local_consumed(self, chan):
        return self.c


def _rank(r, sent=0, consumed=0):
    return SimpleNamespace(ps=SimpleNamespace(pp_rank=r, pp_size=3), pp_flip_counters=_Ctr(sent, consumed),
                           pp_chain_receiver=SimpleNamespace(inbox=collections.deque()), mbs=[None, None, None],
                           chunked_req=None, _pdflip_told_paced_on=True, pdflip_dormant=False)


def test_pp0_frame_carries_the_announcement_only_when_it_holds(monkeypatch):
    monkeypatch.setattr(told, "armed", lambda s: True)
    monkeypatch.setattr(told, "w2c_publishable", lambda s: ["pdflip-34-634"])
    pp0 = _rank(0, sent=41)
    pp0.mbs[1] = object()                                      # the batch just launched in slot 1
    assert w2c.proxy_extra(pp0, 1, None) == {w2c.KEY: 42}
    pp0._w2c_announced = None
    pp0.mbs[2] = object()                                      # another set in flight: no announcement
    assert w2c.proxy_extra(pp0, 1, None) is None
    pp0.mbs[2] = None
    monkeypatch.setattr(told, "w2c_publishable", lambda s: [])
    assert w2c.proxy_extra(pp0, 1, {"a": 1}) == {"a": 1}       # byte-identical frame extras


def test_receiver_pops_the_key_and_the_middle_rank_relays():
    pp1 = _rank(1, sent=17)
    raw = {"hidden_states": "t", w2c.KEY: 42}
    assert w2c.note_frame(pp1, 1, raw) == 42 and w2c.KEY not in raw
    assert w2c.proxy_extra(pp1, 1, None) == {w2c.KEY: 17}
    assert w2c.proxy_extra(pp1, 1, None) is None               # one relay per received announcement


def test_follower_holds_an_announced_frame_until_chain_k_is_absorbed():
    pp1 = _rank(1, consumed=41)
    head = {w2c.KEY: 42}
    assert w2c.follower_defer(pp1, 0, head, budget_s=30.0) is True
    pp1.pp_flip_counters.c = 42
    pp1.pp_chain_receiver.inbox.append(["k"])                  # off the wire but not absorbed by a pass head yet
    assert w2c.follower_defer(pp1, 0, head, budget_s=30.0) is True
    pp1.pp_chain_receiver.inbox.clear()
    assert w2c.follower_defer(pp1, 0, head, budget_s=30.0) is False
    assert w2c.follower_defer(pp1, 0, {"x": 1}, budget_s=30.0) is False


def test_follower_hold_past_the_budget_is_named():
    pp1 = _rank(1, consumed=41)
    head = {w2c.KEY: 42}
    assert w2c.follower_defer(pp1, 0, head, budget_s=0.01) is True
    _time.sleep(0.02)
    with pytest.raises(w2c.AnnouncedChainTimeout, match="#789"):
        w2c.follower_defer(pp1, 0, head, budget_s=0.01)


# ---------------------------------------------------------------------------
# 5. three-rank protocol run

DT = 0.001
F = {0: 1.20, 1: 1.10, 2: 0.80}       # host-blocking forward of the older set per rank (metal 633: 2.4 / 1.2 / 0.4 s)


class _Sim:
    """PP0 -> PP1 -> PP2, a chain wire and a frame wire per hop (no cross-wire ordering), PF acks to PP0, output
    PP2 -> PP0. The ranks run the REAL W2C code (proxy_extra / pp0_take_announced / follower_defer / note_frame);
    PP0's S82 park is the rule of pdflip/pp0_commit_park (first of acks / output)."""

    def __init__(self, *, announce=True, v1=True, chain_delay=None, frame_delay=0.005, budget=30.0):
        self.t = 0.0
        self.v1 = v1
        self.r = {i: _rank(i, sent=40 if i < 2 else 0, consumed=40 if i > 0 else 0) for i in range(3)}
        self.r[2].pp_flip_counters.c = 40
        self.r[1].pp_flip_counters.s = 40
        self.announce = announce
        self.chain_delay = chain_delay or {1: 0.003, 2: 0.003}
        self.frame_delay = frame_delay
        self.budget = budget
        self.chain = {1: collections.deque(), 2: collections.deque()}   # (arrive_t, msg)
        self.frame = {1: collections.deque(), 2: collections.deque()}
        self.acks, self.output_t, self.admit = {}, None, None
        self.row_start, self.told_absorbed, self.computed_keys = {}, {}, []
        self.busy_until = {1: 0.0, 2: 0.0}
        self.pending_frame = {1: None, 2: None}
        self.pp0_state, self.pp0_until = "older", F[0]

    # -- PP0
    def _pp0(self):
        s0 = self.r[0]
        if self.pp0_state == "older" and self.t >= self.pp0_until:
            s0.mbs[0] = "older"
            extra = w2c.proxy_extra(s0, 0, None) if self.announce else None
            self.frame[1].append((self.t + self.frame_delay, dict(extra or {}, rows="older")))
            self.pp0_state, self.pp0_until = "top_k", self.t + 0.004
        elif self.pp0_state == "top_k" and self.t >= self.pp0_until:
            announced = w2c.pp0_take_announced(s0)
            if announced is not None and not self.v1:
                self.pp0_state = "park_before_chain"            # mutant: S82 park before the announced chain
                return
            self._send_chain_k()
        elif self.pp0_state == "park_before_chain":
            if self._acks_in() or self.output_t is not None:
                self._send_chain_k()
        elif self.pp0_state == "park" and self.t >= self.pp0_until:
            if self._acks_in():
                self.admit = ("acks", self.t)
                self.pp0_state = "done"
            elif self.output_t is not None:
                self.admit = ("output", self.t)
                self.pp0_state = "done"

    def _send_chain_k(self):
        s0 = self.r[0]
        s0.pp_flip_counters.s += 1
        w2c.pp0_after_chain_send(s0)
        self.chain[1].append((self.t + self.chain_delay[1], {"told": "young", "k": s0.pp_flip_counters.s}))
        self.pp0_state, self.pp0_until = "park", self.t + 0.004

    def _acks_in(self):
        return all(self.acks.get(i) is not None and self.acks[i] <= self.t for i in (1, 2))

    # -- followers
    def _follower(self, i):
        s = self.r[i]
        if self.t < self.busy_until[i]:
            return
        if self.pending_frame[i] is not None:                  # the row just finished: send it on
            done = self.pending_frame[i]
            self.pending_frame[i] = None
            if i == 1:
                extra = w2c.proxy_extra(s, 0, None)
                self.frame[2].append((self.t + self.frame_delay, dict(extra or {}, rows=done["rows"])))
            else:
                self.output_t = self.t + 0.003
        q = self.chain[i]
        if q and q[0][0] <= self.t:                             # counter-gated chain receive at the pass head
            _, msg = q.popleft()
            s.pp_flip_counters.c += 1
            if msg.get("told"):
                self.told_absorbed[i] = self.t
                self.acks[i] = self.t + 0.002
            if i == 1:
                s.pp_flip_counters.s += 1
                self.chain[2].append((self.t + self.chain_delay[2], msg))
            return
        fq = self.frame[i]
        if fq and fq[0][0] <= self.t:
            head = fq[0][1]
            if w2c.follower_defer(s, 0, head, budget_s=self.budget):
                return
            fq.popleft()
            w2c.note_frame(s, 0, head)
            self.computed_keys.append(sorted(head))
            self.row_start[i] = self.t
            self.busy_until[i] = self.t + F[i]
            self.pending_frame[i] = head

    def run(self, t_max=8.0):
        w2c._clock = lambda: self.t
        try:
            return self._run(t_max)
        finally:
            w2c._clock = _time.monotonic

    def _run(self, t_max):
        while self.t < t_max:
            self._pp0()
            self._follower(1)
            self._follower(2)
            if self.pp0_state == "done" and self.output_t is not None and self.t > self.output_t:
                return self
            self.t += DT
        return self


@pytest.fixture
def _armed(monkeypatch):
    monkeypatch.setattr(told, "armed", lambda s: True)
    monkeypatch.setattr(told, "w2c_publishable", lambda s: ["young"])
    w2c.reset()


def test_three_ranks_ack_before_the_older_row_and_admit_while_it_runs(_armed):
    sim = _Sim().run()
    assert sim.admit is not None and sim.admit[0] == "acks"
    assert sim.output_t is not None and sim.admit[1] < sim.output_t - 1.0     # the younger set runs beside the row
    for i in (1, 2):
        assert sim.told_absorbed[i] < sim.row_start[i]                       # chain k before the older row
    assert all(w2c.KEY not in keys for keys in sim.computed_keys)            # nothing reaches compute with the key


def test_three_ranks_without_announcement_is_todays_serial_pass(_armed, monkeypatch):
    # rc27 16:37:21-27: the frame lands first, PP1 computes the older row, absorbs the told after it
    monkeypatch.setattr(told, "w2c_publishable", lambda s: [])
    sim = _Sim().run()
    assert sim.row_start[1] < sim.told_absorbed[1]
    assert sim.admit is not None and sim.admit[1] >= sim.row_start[1] + F[1]


def test_mutant_park_in_the_announced_pass_deadlocks_until_the_budget(_armed, monkeypatch):
    # review V1 / T3: a follower-dependent wait before chain k -- nobody can ack, nobody can compute
    with pytest.raises(w2c.AnnouncedChainTimeout, match="#789"):
        _Sim(v1=False, budget=2.0).run(t_max=6.0)


def test_mutant_follower_ignores_the_announcement_is_serial(_armed, monkeypatch):
    # review T2: C3 off -> the ack is behind the forward again
    monkeypatch.setattr(w2c, "follower_defer", lambda s, mb, head, budget_s=None: False)
    sim = _Sim(chain_delay={1: 0.010, 2: 0.003}).run()
    assert sim.row_start[1] < sim.told_absorbed[1]
    assert sim.admit[1] >= sim.row_start[1] + F[1]


def test_mutant_middle_rank_without_relay_lets_pp2_overtake_chain_k(_armed, monkeypatch):
    # review V2: the chain and the frame ride two wires with no cross-wire order (#631 row14); a slow PP1->PP2
    # chain hop and no relay -> PP2 computes the older row before it absorbed chain k
    real = w2c.relay_value
    monkeypatch.setattr(w2c, "relay_value", lambda up, sent: None)
    sim = _Sim(chain_delay={1: 0.003, 2: 1.50}).run()
    assert sim.row_start[2] < sim.told_absorbed[2]
    monkeypatch.setattr(w2c, "relay_value", real)
    sim = _Sim(chain_delay={1: 0.003, 2: 1.50}).run()
    assert sim.told_absorbed[2] < sim.row_start[2]


# ---------------------------------------------------------------------------
# 6. loop wiring


def test_scheduler_wiring():
    from flliper.srt.managers import scheduler_pp_mixin as mx

    body = inspect.getsource(mx.SchedulerPPMixin._event_loop_pp_body)
    assert "_pp_w2c.proxy_extra(" in body
    recv = inspect.getsource(mx.SchedulerPPMixin._pp_recv_proxy_tensors)
    assert recv.index("_pp_w2c.note_frame(self, mb_id, raw)") < recv.index("PPProxyTensors(raw)")
    probe = inspect.getsource(mx.SchedulerPPMixin._pp_proxy_frame_pending)
    assert "            if _pp_w2c.follower_defer(self, mb_id, head):\n" in probe
    assert probe.index("_pp_w2c.follower_defer(") < probe.index("THE FRAME MUST NOT OVERTAKE ITS REQUESTS")
    fwd = inspect.getsource(mx.SchedulerPPMixin._pp_forward_and_process_input_requests)
    assert fwd.index("_pp_send_pyobj_to_next_stage(") < fwd.index("_pp_w2c.pp0_after_chain_send(self)")
    pub = inspect.getsource(told.pp0_publish)
    assert pub.index("pp0_take_announced") < pub.index("_commit_park(scheduler, recv_reqs)")
