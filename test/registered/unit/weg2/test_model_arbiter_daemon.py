"""Dual-model: the arbiter side-car around the pure policy.

It reads both fronts' /weg2/state, decides with model_arbiter.decide, and
executes a switch as sleep(A) THEN wake(B) -- never overlapping (ordering
law: BAR1 and PCIe), never waking B when A's sleep failed, and refusing to
act while two fronts claim to be awake.
"""
from sglang.srt.weg2.model_arbiter import ArbiterConfig
from sglang.srt.weg2.model_arbiter_daemon import Arbiter, load_of


class FakeFront:
    def __init__(self, name, model_state, queue=0, outstanding=0, oldest=0.0, log=None,
                 sleep_ok=True, wake_ok=True):
        self.name, self.model_state = name, model_state
        self.queue, self.out, self.oldest = queue, outstanding, oldest
        self.log = log if log is not None else []
        self.sleep_ok, self.wake_ok = sleep_ok, wake_ok

    def state(self):
        return {"model_state": self.model_state, "queue": self.queue,
                "outstanding": {"P": 0, "D": self.out}, "model_oldest_wait_s": self.oldest}

    def model_sleep(self, park):
        self.log.append(("sleep", self.name, park))
        if self.sleep_ok:
            self.model_state = "asleep"
        return {"ok": self.sleep_ok}

    def model_wake(self):
        self.log.append(("wake", self.name))
        if self.wake_ok:
            self.model_state = "awake"
        return {"ok": self.wake_ok}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def mk(a_state="awake", b_state="asleep", **kw):
    log = []
    a = FakeFront("27b", a_state, log=log, **kw.get("a", {}))
    b = FakeFront("nf", b_state, log=log, **kw.get("b", {}))
    clk = Clock()
    arb = Arbiter({"27b": a, "nf": b}, ArbiterConfig(t_max_s=90, t_min_factor=5, t_min_floor_s=10),
                  clock=clk, last_switch_at=0.0, switch_s=6.0)
    return arb, a, b, log, clk


def test_load_counts_queue_and_every_group():
    st = {"queue": 2, "outstanding": {"P": 1, "D": 3}, "model_oldest_wait_s": 4.5}
    ld = load_of(st)
    assert (ld.outstanding, ld.oldest_wait_s) == (6, 4.5)


def test_idle_switch_runs_sleep_then_wake_in_order():
    arb, a, b, log, clk = mk(b={"queue": 1, "oldest": 3.0})
    r = arb.tick()
    assert r["action"] == "switch" and r["reason"] == "idle"
    assert log == [("sleep", "27b", False), ("wake", "nf")]
    assert a.model_state == "asleep" and b.model_state == "awake"
    assert arb.last_switch_at == clk.t


def test_failed_sleep_never_wakes_the_other_model():
    arb, a, b, log, _ = mk(a={"sleep_ok": False}, b={"queue": 1, "oldest": 3.0})
    r = arb.tick()
    assert r["action"] == "error" and "sleep" in r["error"]
    assert log == [("sleep", "27b", False)]
    assert b.model_state == "asleep"


def test_two_awake_fronts_hold_with_an_alarm():
    arb, a, b, log, _ = mk(b_state="awake", b={"queue": 1, "oldest": 99.0})
    r = arb.tick()
    assert r["action"] == "alarm" and "two awake" in r["error"]
    assert log == []


def test_t_max_switch_parks():
    arb, a, b, log, _ = mk(a={"outstanding": 2}, b={"queue": 1, "oldest": 95.0})
    arb.tick()
    assert log[0] == ("sleep", "27b", True)


def test_a_switch_in_progress_is_not_doubled():
    arb, a, b, log, _ = mk(a_state="sleeping", b={"queue": 1, "oldest": 95.0})
    assert arb.tick()["action"] == "hold"
    assert log == []


def test_measured_switch_time_feeds_the_next_dwell():
    arb, a, b, log, clk = mk(b={"queue": 1, "oldest": 3.0})

    def slow_wake():
        clk.t += 12.0
        b.model_state = "awake"
        return {"ok": True}

    b.model_wake = slow_wake
    arb.tick()
    assert arb.switch_s == 12.0
