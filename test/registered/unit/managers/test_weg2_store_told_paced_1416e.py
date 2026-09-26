"""#1416e: the paced told (SGLANG_WEG2_TOLD_PACED) -- no rank busy-waits for a
follower's store read, and the ranks stay equal.

A simulated PP3 ring on the shipped module: PP0 publishes per pass, PP1
absorbs PP0's wire one pass later, PP2 two passes later (the plan lag of the
carrierless form), and every rank runs its admission over its own queue in
every pass. A rank's pass ``k`` plans PP0's pass ``k - pp_rank``; the
group-equality assertion is that every rank admits a rid for the SAME PP0
pass with the SAME told cap. Danger direction pinned: a follower waiting in
admission (the stage stops) -- ``time.sleep`` inside ``admission`` is the
busy-wait and is counted.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.managers import weg2_store_told as m

DT = 0.05  # one pass of wall time


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _TimedTree:
    """Tree double whose store reads terminate at a wall-clock time."""

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


class _Stage:
    def __init__(self, pp_rank, clock, prompts, read_s):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.pp_flip_counters = None
        self.tree_cache = _TimedTree(clock)
        self.waiting_queue = []
        self.prompts = prompts  # rid -> store-hit tokens
        self.read_s = read_s  # rid -> this rank's read duration
        self.admitted = []  # (planned PP0 pass, rid, told cap, credit)

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        hit = self.prompts.get(req.rid, 0)
        if limit_tokens is not None:
            hit = min(hit, int(limit_tokens))
        if hit <= 0:
            return "declined:store_absent"
        self.tree_cache.start_read(req.rid, hit, self.read_s[req.rid])
        return "issued"


def _req(rid):
    return SimpleNamespace(rid=rid, prefetch_deferred=None)


class _Ring:
    """Three stages, the request wire with its per-hop lag, a pass driver."""

    def __init__(self, monkeypatch, prompts, read_s_by_rank, paced=True):
        self.clock = _Clock()
        monkeypatch.setattr(m, "_clock", self.clock)
        if paced:
            monkeypatch.setenv(m.ENV_PACED, "1")
        else:
            monkeypatch.delenv(m.ENV_PACED, raising=False)
        self.stages = [
            _Stage(r, self.clock, prompts, read_s_by_rank[r]) for r in range(3)
        ]
        for s in self.stages:
            assert m.armed(s)
        self.wire = {}  # PP0 pass -> the list PP0 sent
        self.pass_n = 0
        self.sleeps = {1: 0.0, 2: 0.0, 0: 0.0}
        self._stage_in_admission = None

        def _sleep(sec):  # the busy-wait: charge it to the stage and the clock
            self.sleeps[self._stage_in_admission] += sec
            self.clock.t += sec

        monkeypatch.setattr(m.time, "sleep", _sleep)

    def arrive(self, rid):
        for s in self.stages:
            r = _req(rid)
            s.waiting_queue.append(r)
            m.intake(s, r, lambda gate: None)

    def step(self):
        k = self.pass_n
        pp0 = self.stages[0]
        self.wire[k] = m.pp0_publish(pp0, [])
        for s in self.stages[1:]:
            src = k - s.ps.pp_rank
            if src >= 0:
                m.follower_absorb(s, list(self.wire.get(src, [])))
        for s in self.stages:
            self._stage_in_admission = s.ps.pp_rank
            for req in list(s.waiting_queue):
                credit = m.admission(s, req, lambda kind, rid: None)
                if credit is None:
                    continue
                s.waiting_queue.remove(req)
                s.admitted.append(
                    (k - s.ps.pp_rank, req.rid, getattr(req, "_weg2_prefix_cap", None), credit)
                )
        self.pass_n += 1
        self.clock.t += DT

    def run(self, passes):
        for _ in range(passes):
            self.step()

    def plans(self, rid):
        return [[a[:3] for a in s.admitted if a[1] == rid] for s in self.stages]


PROMPTS = {"aaaa-told": 100_000, "bbbb-fresh": 0}


def _read_s(pp0_s, follower_s):
    return {
        0: {"aaaa-told": pp0_s, "bbbb-fresh": 0.0},
        1: {"aaaa-told": follower_s, "bbbb-fresh": 0.0},
        2: {"aaaa-told": follower_s, "bbbb-fresh": 0.0},
    }


def test_paced_second_request_never_waits_and_ranks_agree(monkeypatch):
    """told > 0 for A, B behind it: B is admitted on every rank while A's
    followers read; nobody sleeps; A lands on every rank for the same PP0
    pass with the same told cap."""
    ring = _Ring(monkeypatch, PROMPTS, _read_s(pp0_s=0.5, follower_s=0.8))
    ring.arrive("aaaa-told")
    ring.run(2)
    ring.arrive("bbbb-fresh")
    ring.run(60)  # 3 s of passes
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}  # no busy-wait on any stage
    a, b = ring.plans("aaaa-told"), ring.plans("bbbb-fresh")
    assert all(len(p) == 1 for p in a + b), (a, b)
    assert a[0] == a[1] == a[2]  # same PP0 pass, same told cap on every rank
    assert b[0] == b[1] == b[2]
    assert a[0][0][2] == 100_000
    assert b[0][0][0] < a[0][0][0]  # B was not held behind A
    # A waited the window (max(1.25 x 0.5 s, 1 s per 100k) = 1 s), not a stall
    pp0_pass_a = a[0][0][0]
    assert pp0_pass_a * DT >= 1.0
    credits = [x[3] for s in ring.stages for x in s.admitted if x[1] == "aaaa-told"]
    assert credits == [100_000, 100_000, 100_000]


def test_single_phase_form_busy_waits_on_the_followers(monkeypatch):
    """Characterisation of the risk (switch off): the followers register the
    read when the told arrives and wait for it inside admission -- the stage
    sleeps ~ its read time. This is what the paced form removes."""
    ring = _Ring(monkeypatch, PROMPTS, _read_s(pp0_s=0.5, follower_s=0.8), paced=False)
    ring.arrive("aaaa-told")
    ring.run(2)
    ring.arrive("bbbb-fresh")
    ring.run(60)
    assert ring.sleeps[0] == 0.0
    assert ring.sleeps[1] >= 0.7 and ring.sleeps[2] >= 0.7
    a = ring.plans("aaaa-told")
    assert a[0] == a[1] == a[2]  # consistent -- but bought with a stopped stage


def test_paced_wire_is_read_ahead_then_admit(monkeypatch):
    ring = _Ring(monkeypatch, PROMPTS, _read_s(pp0_s=0.2, follower_s=0.2))
    ring.arrive("aaaa-told")
    ring.run(40)
    kinds = [
        (type(o).__name__, getattr(o, "paced", None))
        for k in sorted(ring.wire)
        for o in ring.wire[k]
    ]
    assert kinds == [("Weg2StoreTold", True), ("Weg2StoreAdmit", None)]
    told_k = next(k for k in ring.wire if ring.wire[k])
    admit_k = max(k for k in ring.wire if ring.wire[k])
    assert admit_k > told_k


def test_paced_told_zero_stays_single_phase(monkeypatch):
    ring = _Ring(monkeypatch, PROMPTS, _read_s(pp0_s=0.0, follower_s=0.0))
    ring.arrive("bbbb-fresh")
    ring.run(4)
    objs = [o for k in sorted(ring.wire) for o in ring.wire[k]]
    assert len(objs) == 1 and isinstance(objs[0], m.Weg2StoreTold)
    assert objs[0].told == 0 and objs[0].paced is False
    b = ring.plans("bbbb-fresh")
    assert b[0] == b[1] == b[2] and len(b[0]) == 1


def test_window_cap_is_the_deadline(monkeypatch):
    """PP0's own read took 20 s: factor x 20 = 25 s, capped at the deadline;
    the Admit goes out at the cap, never later."""
    monkeypatch.setenv(m.ENV_PACE_CAP_S, "1.5")
    assert m.pace_window_s(20.0, 100_000) == 1.5
    monkeypatch.delenv(m.ENV_PACE_CAP_S)
    assert m.pace_window_s(0.1, 100_000) == pytest.approx(1.0)  # told floor
    assert m.pace_window_s(2.0, 1_000) == pytest.approx(2.5)  # factor x own read
    monkeypatch.setenv(m.ENV_PACE_CAP_S, "1.5")
    ring = _Ring(monkeypatch, PROMPTS, _read_s(pp0_s=2.0, follower_s=0.5))
    ring.arrive("aaaa-told")
    ring.run(120)
    a = ring.plans("aaaa-told")
    assert a[0] == a[1] == a[2] and len(a[0]) == 1
    # read-ahead at ~2.0 s, Admit at +1.5 s (cap), not at +2.5 s
    assert 3.4 <= a[0][0][0] * DT <= 3.7
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}


def test_follower_slower_than_the_window_is_the_named_residual(monkeypatch):
    """A follower still reading at the Admit falls into the unchanged bounded
    wait (the residual this form does not remove without a follower->PP0
    channel); past WAIT_CAP_S it refuses by name -- never a silent split."""
    monkeypatch.setenv(m.ENV_PACE_CAP_S, "0.5")
    ring = _Ring(monkeypatch, PROMPTS, _read_s(pp0_s=0.1, follower_s=2.0))
    ring.arrive("aaaa-told")
    ring.run(80)
    a = ring.plans("aaaa-told")
    assert a[0] == a[1] == a[2]
    assert 0 < ring.sleeps[1] < 2.0  # waited only the remainder past the window
    monkeypatch.setattr(m, "WAIT_CAP_S", 0.0)
    ring2 = _Ring(monkeypatch, {"cccc-stuck": 4096}, {
        0: {"cccc-stuck": 0.1}, 1: {"cccc-stuck": 10**6}, 2: {"cccc-stuck": 10**6}})
    ring2.arrive("cccc-stuck")
    with pytest.raises(m.Weg2StoreToldMismatch, match="WAIT EXCEEDED"):
        ring2.run(80)


def test_abort_inside_the_window_publishes_no_admit(monkeypatch):
    ring = _Ring(monkeypatch, PROMPTS, _read_s(pp0_s=0.2, follower_s=0.2))
    ring.arrive("aaaa-told")
    ring.run(8)  # read-ahead out, window (1 s) still running
    assert "aaaa-told" in m._pacing(ring.stages[0])
    for s in ring.stages:
        s.waiting_queue.clear()  # the abort reached every rank
    ring.run(40)
    objs = [type(o).__name__ for k in sorted(ring.wire) for o in ring.wire[k]]
    assert objs == ["Weg2StoreTold"]
    assert m._pacing(ring.stages[0]) == {}
    assert all(not s.admitted for s in ring.stages)


def test_followers_follow_the_wire_not_their_own_env(monkeypatch):
    """PP0 unpaced, followers' env paced: the single-phase told still admits
    on every rank at the same pass (a follower never reads the switch)."""
    clock = _Clock()
    monkeypatch.setattr(m, "_clock", clock)
    pp0 = _Stage(0, clock, PROMPTS, _read_s(0.0, 0.0)[0])
    monkeypatch.delenv(m.ENV_PACED, raising=False)
    assert m.armed(pp0) and pp0._weg2_told_paced_on is False
    monkeypatch.setenv(m.ENV_PACED, "1")
    pp1 = _Stage(1, clock, PROMPTS, _read_s(0.0, 0.0)[1])
    assert m.armed(pp1)
    r0, r1 = _req("bbbb-fresh"), _req("bbbb-fresh")
    pp0.waiting_queue.append(r0)
    pp1.waiting_queue.append(r1)
    m.intake(pp0, r0, lambda g: None)
    m.intake(pp1, r1, lambda g: None)
    wire = m.pp0_publish(pp0, [])
    assert [getattr(o, "paced", None) for o in wire] == [False]
    m.follower_absorb(pp1, list(wire))
    assert m.admission(pp0, r0, lambda k, r: None) == 0
    assert m.admission(pp1, r1, lambda k, r: None) == 0


def test_admit_before_the_request_reaches_the_follower_registers_at_intake(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(m, "_clock", clock)
    s = _Stage(2, clock, PROMPTS, _read_s(0.0, 0.0)[2])
    assert m.armed(s)
    m.follower_absorb(s, [m.Weg2StoreTold("aaaa-told", 100_000, paced=True)])
    r = _req("aaaa-told")
    s.waiting_queue.append(r)
    assert m.intake(s, r, lambda g: None) == "issued"  # read-ahead found
    assert m.admission(s, r, lambda k, x: None) is None  # no Admit yet
    m.follower_absorb(s, [m.Weg2StoreAdmit("aaaa-told", 100_000)])
    assert m.admission(s, r, lambda k, x: None) == 100_000
