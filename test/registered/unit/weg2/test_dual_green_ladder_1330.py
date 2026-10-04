# SPDX-License-Identifier: Apache-2.0
"""Item 1330: the green-context ladder of P (weg2/dual_green.py), dynamic and stepwise -- DESK tests, no GPU.

The CUDA side sits behind ``GreenBackend``; these tests drive the same ladder / controller / wire / launch code
with a FAKE backend (driver rounding rule, scripted probe timings, scripted create failures). Pinned:

  * LADDER: the SM numbers come from the driver's group sizes (5090: 8-SM granularity 170 -> 128/88/48; 3080:
    2-SM granularity 68 -> 52/34/18, the metal probe's numbers), a rung whose mask does not bite or that cannot
    be created is NAMED and served by the fallback (counters in the marker), a backend that cannot even answer
    ``info`` never raises.
  * STAGE AUTOMATON (section 8): D empty -> stage 0 at once; entry stage by the open loop (the more bs, the
    DEEPER); closed loop on the measured D round (descend at once, two stages when far over, ascend only after
    calm + dwell + preview); hysteresis; starvation clamp; flaps counted.
  * WIRE / RANK UNIFORMITY: three stages through the REAL ``_pp_forward_and_process_input_requests``: PP0 stamps
    ONE stage on the request wire, the followers relay it, apply it and dispatch without it; no follower reads
    the ctl file.
  * LAUNCH: stream order of the forward (the green stream waits for forward_stream, forward_stream waits for the
    green stream), the stage changes only at a forward boundary, graphs go eager below 100 %.
  * HOLD: observer by default; armed: bounded, cooled down, released at once when D is empty, starvation-exempt.
  * "FLIP / NF / INT8 UNCHANGED": every wrong gate -> no actuator, no import of the CUDA path, no wire object,
    launcher argv/env byte-equal with the switch off, the plain Decision's ctl line byte-equal.
  * MUTANTS: the source of dual_green is mutated (gate out, dwell out, stage mapping wrong, wire only PP0, D-empty
    jump out, starvation exception out); each mutant turns its own check RED.
"""
from __future__ import annotations

import contextlib
import inspect
import math
import os
import pickle
import sys
import tempfile
import types
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import dual_green as G  # noqa: E402
from sglang.srt.weg2 import dual_share as S  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

DUAL_P = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "4096"}
LADDER_ENV = {S.GREEN_LADDER_ENV: "1", S.ACT_ENV: "green,duty", S.CTL_ENV: "/dev/shm/x.ctl"}
FRACTIONS = (1.0, 0.75, 0.5, 0.25)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    G._reset_state_for_tests()
    for k in list(DUAL_P) + [S.GREEN_LADDER_ENV, S.ACT_ENV, S.CTL_ENV, G.HOLD_ENV, G.PROBE_ENV, S.DUTY_ENV]:
        monkeypatch.delenv(k, raising=False)
    yield
    G._reset_state_for_tests()


# --------------------------------------------------------------------------------------------- the fake CUDA
class FakeStream:
    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other.name)
        LOG.append((self.name, "waits", other.name))

    def __repr__(self):
        return f"<{self.name}>"


LOG: list = []


class FakeBackend(G.GreenBackend):
    """The driver's documented rule: groups of equal size, rounded UP to the granularity."""

    def __init__(self, total, gran, *, name="fake", fail_create=(), mask_bites=True, info_fails=False, probe_fails=False):
        self.total, self.gran, self.name = total, gran, name
        self.fail_create, self.mask_bites, self.info_fails, self.probe_fails = set(fail_create), mask_bites, info_fails, probe_fails
        self.created = []

    def info(self):
        if self.info_fails:
            raise G.GreenError("cuDeviceGetDevResource -> CUDA_ERROR_NOT_SUPPORTED")
        return G.DevInfo(0, self.name, "sm120", f"GPU-{self.name}", self.total, self.gran, "cuda13020")

    def split_sizes(self, want):
        size = int(math.ceil(max(1, want) / self.gran) * self.gran)
        n = self.total // size
        return [size] * n if n >= 1 else []

    def create(self, want):
        if want in self.fail_create:
            raise G.GreenError("cuGreenCtxCreate -> CUDA_ERROR_NOT_SUPPORTED")
        sm = self.split_sizes(want)[0]
        st = FakeStream(f"gc{sm}")
        self.created.append((want, sm))
        return sm, st

    def time_stream(self, stream):
        if self.probe_fails:
            raise RuntimeError("probe boom")
        if stream is None:
            return 10.0
        sm = int(stream.name[2:])
        return 10.0 * (self.total / sm) * 0.9 if self.mask_bites else 10.2


def ladder_of(backend, **kw):
    warns, logs = [], []
    ld = G.GreenLadder(backend, FRACTIONS, log=logs.append, warn=warns.append, **kw).build()
    return ld, warns, logs


# --------------------------------------------------------------------------------------------------- ladder
def test_ladder_sm_numbers_are_the_drivers_on_both_card_types():
    ld5, w5, _ = ladder_of(FakeBackend(170, 8, name="5090"))
    ld3, w3, _ = ladder_of(FakeBackend(68, 2, name="3080"))
    assert {i: r.sm for i, r in ld5.rungs.items()} == {1: 128, 2: 88, 3: 48}       # metal probe c1: 128/88/48
    assert {i: r.sm for i, r in ld3.rungs.items()} == {1: 52, 2: 34, 3: 18}        # metal probe c0/c2
    assert w5 == [] and w3 == [] and ld5.failed == {} and ld3.failed == {}
    assert ld5.counters["created"] == 3 and ld5.counters["probe_run"] == 3
    assert "rungs=[0.75:128,0.50:88,0.25:48]" in ld5.marker() and "created=3" in ld5.marker()


def test_the_fraction_is_the_shared_quantity_each_card_maps_it_to_its_own_ladder():
    ld5, _, _ = ladder_of(FakeBackend(170, 8))
    ld3, _, _ = ladder_of(FakeBackend(68, 2))
    for f, sm5, sm3 in ((0.75, 128, 52), (0.5, 88, 34), (0.25, 48, 18)):
        assert ld5.entry(f).sm == sm5 and ld3.entry(f).sm == sm3
    assert ld5.entry(1.0) is None and ld5.serves(1.0) and ld5.serves(0.5)


def test_a_mask_that_does_not_bite_is_named_and_dropped():
    ld, warns, _ = ladder_of(FakeBackend(170, 8, mask_bites=False))
    assert ld.rungs == {} and sorted(ld.failed) == [1, 2, 3] and ld.counters["probe_rejected"] == 3
    assert warns and all(w.startswith(S.FALLBACK + " mech=green") and "does not bite" in w for w in warns)
    assert not ld.serves(0.5) and ld.serves(1.0)


def test_a_rung_that_cannot_be_created_is_named_the_others_still_serve():
    ld, warns, _ = ladder_of(FakeBackend(170, 8, fail_create=(85,)))
    assert sorted(ld.rungs) == [1, 3] and 2 in ld.failed and ld.counters["create_failed"] == 1
    assert any("rung 2" in w and "cuGreenCtxCreate" in w and "duty/chunk" in w for w in warns)
    assert ld.serves(0.75) and not ld.serves(0.5) and ld.serves(0.25)
    assert "failed_rungs=[2]" in ld.marker()


def test_a_backend_that_cannot_answer_never_raises_and_names_it():
    ld, warns, _ = ladder_of(FakeBackend(170, 8, info_fails=True))
    assert ld.rungs == {} and sorted(ld.failed) == [1, 2, 3]
    assert len(warns) == 1 and "CUDA_ERROR_NOT_SUPPORTED" in warns[0] and "ladder not built" in warns[0]
    assert "rungs=[]" in ld.marker()


def test_a_failing_probe_baseline_keeps_the_rungs_unprobed_and_named():
    ld, warns, _ = ladder_of(FakeBackend(170, 8, probe_fails=True))
    assert sorted(ld.rungs) == [1, 2, 3] and any("probe baseline failed" in w for w in warns)


def test_two_fractions_that_round_to_one_group_share_one_context():
    be = FakeBackend(16, 8)                                  # 0.5 -> 8, 0.25 -> 4 -> rounded up to 8: the same group
    ld, _, _ = ladder_of(be)
    assert len(be.created) == 2 and ld.rungs[1].sm == 16 and ld.rungs[2].sm == ld.rungs[3].sm == 8
    assert ld.rungs[2].stream is ld.rungs[3].stream


# ------------------------------------------------------------------------------------------ gate / maybe_arm
def _sched(rank=0, size=3):
    return SimpleNamespace(ps=SimpleNamespace(pp_rank=rank, pp_size=size))


def test_maybe_arm_is_none_off_every_wrong_gate_and_never_touches_the_backend():
    class Boom(G.GreenBackend):
        def __getattribute__(self, n):
            raise AssertionError("the backend was touched")

    wrong = [{}, dict(LADDER_ENV), {**DUAL_P}, {**DUAL_P, S.GREEN_LADDER_ENV: "1"},                   # no actuator / no ctl
             {**DUAL_P, S.GREEN_LADDER_ENV: "0", S.ACT_ENV: "green", S.CTL_ENV: "/x"},                # switch 0
             {**DUAL_P, S.GREEN_LADDER_ENV: "1", S.ACT_ENV: "chunk,duty", S.CTL_ENV: "/x"},            # no green actuator
             {**LADDER_ENV, "SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "1"},
             {**LADDER_ENV, "SGLANG_WEG2_DUAL_LAYOUT": "0", "SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "1"},
             {**LADDER_ENV, "SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_DUAL_LAYOUT": "1"}]                  # uncapped P
    for env in wrong:
        warns = []
        assert G.armed(env) is False, env
        assert G.maybe_arm(_sched(), env, backend=Boom(), warn=warns.append) is None, env
        if G.ladder_switch(env):                       # a set switch that cannot arm is NAMED, never silent
            assert len(warns) == 1 and warns[0].startswith(S.FALLBACK + " mech=green") and "gate is off" in warns[0], env
        else:
            assert warns == [], env
    assert G.force_eager() is False and S.green_serves(1) is False


def test_maybe_arm_on_the_gate_returns_an_actuator_even_when_everything_fails():
    env = {**DUAL_P, **LADDER_ENV}
    logs, warns = [], []
    act = G.maybe_arm(_sched(1), env, backend=FakeBackend(170, 8, info_fails=True), log=logs.append, warn=warns.append)
    assert act is not None and act.ladder.healthy_count == 0 and not act.first
    assert any("no rung served" in w for w in warns)
    assert S.green_serves(1) is False                           # the fallback actuators keep throttling
    act2 = G.maybe_arm(_sched(1), env, backend=G._DeadBackend("no libcuda"), log=logs.append, warn=warns.append)
    assert act2 is not None and act2.ladder.healthy_count == 0


def test_the_real_ctypes_backend_without_a_gpu_is_a_named_fallback_not_a_crash():
    """This box hides every GPU (CUDA_VISIBLE_DEVICES empty): the real backend cannot be made."""
    env = {**DUAL_P, **LADDER_ENV}
    logs, warns = [], []
    act = G.maybe_arm(_sched(2), env, log=logs.append, warn=warns.append)          # backend=None -> CtypesBackend()
    assert act is not None and act.ladder.healthy_count == 0 and act.ladder.info is None
    assert any(w.startswith(S.FALLBACK + " mech=green") for w in warns)
    assert any("serving=0" in l for l in logs) and S.green_serves(1) is False and G.force_eager() is False
    a = act
    sched = _launch_sched()
    a.apply(G.Weg2DualGreenRung(1, 2, 500000))
    ctx, stream = a.pick(sched)                                                       # every stage: the primary stream
    assert stream is None and a.fallback_forwards == 1


def test_maybe_arm_registers_serving_and_the_fallback_actuators_step_aside():
    env = {**DUAL_P, **LADDER_ENV}
    act = G.maybe_arm(_sched(0), env, backend=FakeBackend(170, 8), log=lambda m: None, warn=lambda m: None)
    assert act is not None and act.ladder.healthy_count == 3 and S.green_serves(1) and S.green_serves(3)
    # a chunk cap / duty at a served rung does nothing; at rung 0 nothing either
    ctl = tempfile.mktemp()
    w = S.CtlWriter(ctl, heartbeat_s=0.0)
    d = S.Decision("dynamic", 2, 1, 2, 0.5, None, 1, 4, 0.0, None, None, "x", True, False, 0, 0.0)
    w.update(d)
    rd = S.CtlReader(ctl)
    assert S.ChunkCap(rd, 1024, 64, (512,)).cap(1024) == 1024
    duty = S.ShareDuty(rd, sleep=lambda s: None)
    duty.after_forward(0.2)
    assert duty.pause_s() == 0.0
    S.set_green_serves(None)                                    # without the ladder: the base behaviour (RED -> GREEN pair)
    assert S.ChunkCap(S.CtlReader(ctl), 1024, 64, (512,)).cap(1024) == 512
    duty2 = S.ShareDuty(S.CtlReader(ctl), sleep=lambda s: None)
    duty2.after_forward(0.2)
    assert duty2.pause_s() > 0.0
    os.unlink(ctl)


# -------------------------------------------------------------------------------------------- the controller
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def ctrl(rmin=0.0, mode="dynamic", gcfg=None, **cfgkw):
    clk = Clock()
    cfg = S.ShareConfig(d_min_rate_tps=rmin, **cfgkw)
    return G.GreenController(cfg, mode, gcfg or G.GreenConfig(), clock=clk), clk


def obs(c, clk, *, b, round_ms=None, q=0.0, tau=None, age=0.0, dt=0.0, p_idle=False):
    clk.t += dt
    rate = None if round_ms is None else 1000.0 * c.gcfg.accept_len / round_ms
    prate = None if tau is None else (q / tau if q > 0 else 1000.0)
    return c.observe(q_tokens=q, b=b, seats=6, p_rate_tps=prate, d_rate_tps=rate, oldest_age_s=age, p_idle=p_idle)


def test_d_empty_is_stage_0_at_once_from_any_depth():
    c, clk = ctrl(rmin=25.0)
    obs(c, clk, b=6, dt=1.0)
    assert c.rung >= 2
    d = obs(c, clk, b=0, dt=0.01)                     # 10 ms later, far inside every dwell
    assert d.rung == 0 and d.reason == "d_idle" and d.changed and d.adj == 0


def test_entry_by_the_open_loop_the_more_bs_the_deeper_with_a_target():
    stages = []
    for bs in (1, 3, 6):
        c, clk = ctrl(rmin=25.0)                       # target 100 ms per round (accept_len 2.5)
        d = obs(c, clk, b=bs, dt=1.0)
        stages.append(d.rung)
        assert d.reason.endswith("+entry") and d.changed
    assert stages == [1, 2, 2]                         # bs1: 2.8*28 = 78 ms (75 %); bs3: 1.5*40 = 60 (50 %); bs6: 1.5*57 = 85
    c, clk = ctrl(rmin=60.0)                           # a hard target (42 ms): bs1 needs 25 % (1.24*28 = 35), bs6 fits nowhere
    assert obs(c, clk, b=1, dt=1.0).rung == 3
    c, clk = ctrl(rmin=60.0)
    d = obs(c, clk, b=6, dt=1.0)
    assert d.rung == 3 and "none_fits" in d.reason


def test_entry_by_the_table_without_a_target_bs_classes_and_tau_class():
    for bs, low, high in ((1, 1, 0), (2, 1, 0), (3, 2, 1), (4, 2, 1), (5, 3, 2), (6, 3, 2)):
        c, clk = ctrl()
        assert obs(c, clk, b=bs, dt=1.0, q=1000.0, tau=0.5).rung == low, (bs, "tau low")
        c, clk = ctrl()
        assert obs(c, clk, b=bs, dt=1.0, q=30000.0, tau=30.0).rung == high, (bs, "tau high")
    c, clk = ctrl()
    assert obs(c, clk, b=1, dt=1.0).rung == 1          # tau unknown = the low column


def test_static_modes_keep_their_rung_and_p_stays_full():
    c, clk = ctrl(mode="balanced")
    assert obs(c, clk, b=2, dt=1.0).rung == 2
    c, clk = ctrl(mode="d")
    assert obs(c, clk, b=2, dt=1.0).rung == 3 and obs(c, clk, b=0, dt=0.01).rung == 0
    c, clk = ctrl(mode="p")
    assert obs(c, clk, b=5, dt=1.0).rung == 0


def _entered(rmin=25.0, bs=1):
    c, clk = ctrl(rmin=rmin)
    d = obs(c, clk, b=bs, dt=1.0)
    return c, clk, d


def test_overshoot_descends_one_stage_after_the_dwell_and_two_when_far_over():
    c, clk, d = _entered()
    assert d.rung == 1 and d.target_ms == 100.0
    d = obs(c, clk, b=1, round_ms=150.0, dt=0.2)       # over target, but 0.2 s since the change
    assert d.rung == 1
    d = obs(c, clk, b=1, round_ms=150.0, dt=0.9)       # 1.1 s since the change (desc_min_s 1.0)
    assert d.rung == 2 and "d_round_over" in d.reason and d.adj == 1
    d = obs(c, clk, b=1, round_ms=260.0, dt=1.1)       # x2.6 the target: two stages, clamped at the deepest
    assert d.rung == 3 and d.adj == 2
    d = obs(c, clk, b=1, round_ms=400.0, dt=1.0)       # nothing deeper than p_min_share allows
    assert d.rung == 3


def test_calm_ascends_one_stage_only_after_calm_dwell_and_a_passing_preview():
    c, clk, d = _entered()
    obs(c, clk, b=1, round_ms=150.0, dt=1.1)           # -> rung 2
    assert c.rung == 2
    d = obs(c, clk, b=1, round_ms=50.0, dt=0.5)        # calm starts
    assert d.rung == 2
    d = obs(c, clk, b=1, round_ms=50.0, dt=1.0)        # 1.0 s calm: not yet 1.5
    assert d.rung == 2
    d = obs(c, clk, b=1, round_ms=50.0, dt=1.0)        # 2.0 s calm, 2.6 s dwell: preview 2.8*(50/28/1.5)*28 = 93.3 <= 100
    assert d.rung == 1 and "d_round_calm" in d.reason and "preview_ok" in d.reason and d.adj == 0


def test_calm_with_a_failing_preview_stays_and_says_so():
    c, clk, d = _entered()
    obs(c, clk, b=1, round_ms=150.0, dt=1.1)
    assert c.rung == 2
    obs(c, clk, b=1, round_ms=70.0, dt=0.5)            # ratio exactly 0.7 = calm, but the stage-1 preview is 131 ms
    d = obs(c, clk, b=1, round_ms=70.0, dt=2.5)
    assert d.rung == 2 and "calm_preview_blocks" in d.reason


def test_a_round_between_the_calm_and_the_over_line_is_the_dead_band_nothing_moves():
    c, clk, d = _entered()
    for _ in range(12):
        d = obs(c, clk, b=1, round_ms=85.0, dt=0.5)    # 0.85 x target: neither over nor calm
        assert d.rung == 1 and not d.changed
    assert c.flaps == 0


def test_flaps_are_counted_when_a_change_reverses_inside_the_window():
    c, clk, d = _entered()
    obs(c, clk, b=1, round_ms=150.0, dt=1.1)           # down
    assert c.rung == 2 and c.flaps == 0
    obs(c, clk, b=1, round_ms=50.0, dt=0.5)
    obs(c, clk, b=1, round_ms=50.0, dt=2.5)            # up inside flap_window_s (10 s)
    assert c.rung == 1 and c.flaps == 1


def test_a_bs_change_while_busy_moves_by_the_dwell_rule_one_stage_per_decision():
    c, clk, d = _entered(bs=1)                         # stage 1
    d = obs(c, clk, b=6, dt=0.2)                       # start is now 2 (bs6) but 0.2 s < dwell_to_d 0.5
    assert d.rung == 1 and d.held and "dwell_hold" in d.reason
    d = obs(c, clk, b=6, dt=0.5)
    assert d.rung == 2 and not d.held


def test_the_entry_after_idle_starts_fresh_the_closed_loop_state_is_gone():
    c, clk, d = _entered()
    obs(c, clk, b=1, round_ms=260.0, dt=1.1)
    assert c.adj > 0
    obs(c, clk, b=0, dt=0.1)
    d = obs(c, clk, b=1, dt=0.1)
    assert d.rung == 1 and d.adj == 0 and "+entry" in d.reason


def test_the_starvation_clamp_wins_and_p_min_share_caps_the_depth():
    c, clk = ctrl(rmin=60.0)
    d = obs(c, clk, b=1, dt=1.0)
    assert d.rung == 3
    d = obs(c, clk, b=1, dt=0.1, age=61.0)
    assert d.rung <= 1 and "starve" in d.reason and d.starve
    c2, clk2 = ctrl(rmin=60.0, p_min_share=0.5)
    d = obs(c2, clk2, b=1, dt=1.0)
    assert c2.cfg.max_rung() == 2 and d.rung == 2                       # never deeper than p_min_share allows


def test_tsolo_learns_from_d_alone_and_only_then_replaces_the_default():
    c, clk = ctrl(rmin=25.0)
    assert c.tsolo.get(2)[1] == "default"
    for _ in range(3):
        obs(c, clk, b=2, round_ms=36.0, dt=0.3, p_idle=True)
    ms, src = c.tsolo.get(2)
    assert src == "learned" and abs(ms - 36.0) < 1e-6
    n = c.tsolo.learned[2][1]
    obs(c, clk, b=2, round_ms=500.0, dt=0.3, p_idle=False)    # a round measured WITH P is not the solo round
    assert c.tsolo.learned[2][1] == n
    assert c.tsolo.get(1)[0] == 28.0 and c.tsolo.get(6)[0] == 57.0 and 28.0 < c.tsolo.get(4)[0] < 57.0


def test_decision_line_and_pstufe_line_carry_the_spec_fields_and_the_ctl_line_the_starve_flag():
    c, clk, d = _entered()
    c.feed_pobs({0: {"arena_ppm": "930000", "would_hold": "0", "hold": "0", "sm": "128"}, 1: {"sm": "52"}, 2: {"sm": "52"}})
    line = G.pstufe_line(d, c.pobs_view())
    for f in ("P-STUFE", "bs=1/6", "kv=n/a", "arena=0.930", "pending=", "tau=", "d_round_ms=", "target_ms=100",
              "stufe=0->1", "f=0.75", "sm=0:128,1:52,2:52", "would_hold=0", "reason=", "flaps=0"):
        assert f in line, (f, line)
    assert S.format_ctl(7, d).strip().endswith("starve=0") and "green" not in S.format_ctl(7, d)
    plain = S.Decision("dynamic", 1, 0, 1, 0.75, None, 1, 6, 0.0, None, None, "x", True, False, 0, 0.0)
    assert S.format_ctl(7, plain) == "v1 seq=7 mode=dynamic rung=1 f=0.7500 b=1 seats=6 q=0 tau_ms=-1 r_x100=-1\n"


def test_the_ctl_roundtrip_carries_starve_to_the_p_side():
    c, clk, d = _entered()
    d2 = obs(c, clk, b=1, dt=0.1, age=61.0)
    path = tempfile.mktemp()
    S.CtlWriter(path, heartbeat_s=0.0).update(d2)
    st = S.CtlReader(path).read()
    assert st.ok and st.starve is True and st.rung == d2.rung
    os.unlink(path)


# ------------------------------------------------------------------------------------- front glue (FrontShare)
def test_front_share_with_the_ladder_uses_the_green_controller_and_logs_pstufe(monkeypatch):
    logs = []
    ctl = tempfile.mktemp()
    fs = S.FrontShare.from_args(ctl=ctl, mode="dynamic", actuators="green,duty", d_min_rate_tps=25.0, p_min_share=0.25,
                                env={}, log=logs.append, green_ladder="on")
    assert isinstance(fs.ctrl, G.GreenController) and fs.green == "on" and fs.ctrl.cfg.tick_s == G.GreenConfig().tick_s
    assert fs.d_rate.window_s == 1.0 and G.GreenConfig().desc_min_s >= 1.0
    d = fs.tick(queue=[], p_outstanding={}, d_outstanding={"r1": 1.0}, seats=6)
    assert d.rung == 1 and fs.snapshot()["green"] == "on"
    assert any(l.startswith("DUAL-SHARE mode=") for l in logs) and any(l.startswith("P-STUFE ") for l in logs)
    assert S.CtlReader(ctl).read().rung == 1
    os.unlink(ctl)
    off = S.FrontShare.from_args(ctl=ctl, mode="dynamic", actuators="green,duty", d_min_rate_tps=25.0, p_min_share=0.25,
                                 env={}, log=logs.append, green_ladder="off")
    assert type(off.ctrl) is S.ShareController and off.green is None and "green" not in off.snapshot()
    nogreen = S.FrontShare.from_args(ctl=ctl, mode="dynamic", actuators="chunk", d_min_rate_tps=25.0, p_min_share=0.25,
                                     env={}, log=logs.append, green_ladder="on")
    assert type(nogreen.ctrl) is S.ShareController                      # no 'green' actuator: the base controller


# --------------------------------------------------------------------------------------------- the actuator
def make_actuator(rank, ctl_state, *, backend=None, hold=None, clock=None, size=3, pobs=None):
    be = backend or FakeBackend(170, 8)
    ld = G.GreenLadder(be, FRACTIONS, log=lambda m: None, warn=lambda m: None).build()
    box = {"st": ctl_state}

    class R:                                        # a reader the test steers (one reader: PP0's)
        path = "/x"
        reads = 0

        def read(self):
            R.reads += 1
            return box["st"]

    logs = []
    kw = {"clock": clock} if clock else {}
    act = G.GreenActuator(ld, R(), pp_rank=rank, pp_size=size, hold=hold, pobs=pobs, log=logs.append, **kw)
    act.logs, act.box, act.reader_cls = logs, box, R
    return act


def st(rung, f, busy=True, starve=False):
    return S.CtlState(rung=rung, fraction=f, d_busy=busy, mode="dynamic", seq=1, ok=True, starve=starve)


def test_pp0_stamps_on_change_and_heartbeat_followers_absorb_and_the_list_is_clean():
    t = [0.0]
    a0 = make_actuator(0, st(1, 0.75), clock=lambda: t[0])
    a1 = make_actuator(1, st(0, 1.0))
    wire, cmd = a0.pp0_stamp(["req"])
    assert cmd == G.Weg2DualGreenRung(1, 1, 750000) and wire == ["req", cmd] and a0.rung == 1 and a0.f == 0.75
    wire2, cmd2 = a0.pp0_stamp(["req2"])                          # unchanged inside the heartbeat: nothing on the wire
    assert cmd2 is None and wire2 == ["req2"]
    t[0] = 1.5
    _, hb = a0.pp0_stamp([])
    assert hb is not None and hb.seq == 2 and hb.rung == 1        # the heartbeat re-asserts (new seq)
    a0.box["st"] = st(2, 0.5)
    t[0] = 1.6
    _, c3 = a0.pp0_stamp([])
    assert c3.rung == 2 and c3.seq == 3
    rest = a1.follower_absorb(pickle.loads(pickle.dumps(wire)))   # the wire is pickled between stages
    assert rest == ["req"] and a1.rung == 1 and a1.f == 0.75 and a1.seq == 1
    assert a1.follower_absorb(["x"]) == ["x"] and a1.follower_absorb([]) == []
    assert G.without_green_rung(wire) == ["req"]


def test_a_follower_never_reads_the_ctl_file_and_never_stamps():
    a1 = make_actuator(1, st(3, 0.25))
    wire, cmd = a1.pp0_stamp(["req"])
    assert cmd is None and wire == ["req"] and a1.reader_cls.reads == 0
    a0solo = make_actuator(0, st(3, 0.25), size=1)               # a PP group of one has no wire
    assert a0solo.pp0_stamp([])[1] is None


def test_the_stage_changes_only_at_a_forward_boundary():
    a = make_actuator(1, st(0, 1.0))
    sched = _launch_sched()
    ctx, stream = a.pick(sched)
    assert stream is None and a.active == 0
    a.apply(G.Weg2DualGreenRung(5, 2, 500000))                    # absorbed between two forwards
    assert a.rung == 2 and a.active == 0 and G.force_eager() is False   # the launched forward keeps its stream
    ctx, stream = a.pick(sched)                                   # the next forward boundary
    assert a.active == 2 and stream.name == "gc88" and G.force_eager() is True
    assert any("P rung 0->2" in l and "sm_real=88" in l and "eager=1" in l and "seq=5" in l for l in a.logs)
    a.apply(G.Weg2DualGreenRung(6, 0, 1000000))
    ctx, stream = a.pick(sched)
    assert stream is None and a.active == 0 and G.force_eager() is False   # back up: primary stream, graphs allowed
    assert a.switches == 2 and a.by_rung == {0: 2, 2: 1}


def test_the_status_line_with_the_counters_comes_every_30_seconds():
    t = [0.0]
    a = make_actuator(1, st(0, 1.0), clock=lambda: t[0])
    sched = _launch_sched()
    a.pick(sched)
    assert not any("P status" in l for l in a.logs)
    t[0] = 31.0
    a.apply(G.Weg2DualGreenRung(1, 2, 500000))
    a.pick(sched)
    lines = [l for l in a.logs if "P status" in l]
    assert len(lines) == 1 and "forwards=2" in lines[0] and "switches=1" in lines[0] and "by_rung={0: 1, 2: 1}" in lines[0]
    a.pick(sched)
    assert len([l for l in a.logs if "P status" in l]) == 1


def test_a_stage_the_ladder_cannot_serve_runs_on_the_primary_stream_and_is_counted():
    a = make_actuator(1, st(0, 1.0), backend=FakeBackend(170, 8, fail_create=(85,)))
    sched = _launch_sched()
    a.pick(sched)
    a.apply(G.Weg2DualGreenRung(2, 2, 500000))
    ctx, stream = a.pick(sched)
    assert stream is None and a.active == 0 and a.fallback_forwards == 1 and G.force_eager() is False
    assert not a.serves_rung(2) and a.serves_rung(1) and a.serves_rung(0)


# ---------------------------------------------------------------------------------- the launch hook (real code)
def _launch_sched():
    fs, ss = FakeStream("forward"), FakeStream("schedule")

    class Dev:
        @staticmethod
        def stream(s):
            @contextlib.contextmanager
            def cm():
                LOG.append(("enter", s.name))
                yield
                LOG.append(("exit", s.name))
            return cm()

        class Event:
            def record(self, s):
                LOG.append(("event.record",))

        @staticmethod
        def current_stream():
            return FakeStream("cur")

    return SimpleNamespace(forward_stream=fs, schedule_stream=ss, forward_stream_ctx=Dev.stream(fs), device_module=Dev,
                           pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False))


def _real_launch(sched, actuator, batch):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    sched._dual_green = actuator
    sched._pp_bubble_meter = lambda: None
    sched.ps = SimpleNamespace(pp_rank=1)
    sched.run_batch = lambda b, p: LOG.append(("run_batch",)) or SimpleNamespace(can_run_cuda_graph=False)
    sched.forward_ct = 0
    return SchedulerPPMixin._pp_launch_batch(sched, 0, batch, None, [None], __import__("collections").deque())


def test_launch_stream_order_and_the_rung_on_the_batch(monkeypatch):
    LOG.clear()
    a = make_actuator(1, st(0, 1.0))
    sched = _launch_sched()
    a.pick(sched)                                                    # prime (rung 0)
    LOG.clear()
    a.apply(G.Weg2DualGreenRung(2, 2, 500000))
    batch = SimpleNamespace(reqs=[])
    _real_launch(sched, a, batch)
    seq = [e for e in LOG if e[0] != "event.record"]
    i = lambda x: seq.index(x)
    assert i(("enter", "gc88")) < i(("forward", "waits", "schedule")) < i(("gc88", "waits", "forward")) < i(("run_batch",))
    assert i(("run_batch",)) < i(("forward", "waits", "gc88")) < i(("exit", "gc88"))
    assert batch._dual_green_rung == 2                              # the Prefill rank batch line's rung=
    LOG.clear()
    a.apply(G.Weg2DualGreenRung(3, 0, 1000000))
    batch2 = SimpleNamespace(reqs=[])
    _real_launch(sched, a, batch2)
    assert ("enter", "forward") in LOG and not any(e[0] == "gc88" for e in LOG) and batch2._dual_green_rung == 0
    assert ("forward", "waits", "schedule") in LOG


def test_launch_without_the_ladder_is_the_base_path(monkeypatch):
    LOG.clear()
    sched = _launch_sched()
    batch = SimpleNamespace(reqs=[])
    _real_launch(sched, None, batch)
    assert not hasattr(batch, "_dual_green_rung")
    assert [e for e in LOG if e[0] != "event.record"] == [("enter", "forward"), ("forward", "waits", "schedule"),
                                                          ("run_batch",), ("exit", "forward")]


def test_the_graph_runner_asks_the_ladder_first_and_the_prefill_line_carries_the_rung():
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner as R

    src = inspect.getsource(R.can_run_graph)
    assert src.index("_dgr_force_eager()") < src.index("self._full_graph_ineligible_reason")
    assert '"green_rung"' in src
    G._STATE["eager"] = True
    assert G.force_eager() is True
    G._STATE["eager"] = False


def _prefill_lines(caplog, records):
    """Drive the real RankPrefillLog: ``records`` = [(rung or None, timed)], one duration each when timed."""
    import logging

    from sglang.srt.managers.scheduler_components import metrics_reporter as M

    class T:
        def _report(self):
            pass

    rpl = M.RankPrefillLog()
    rpl.timer = T()
    with caplog.at_level(logging.INFO, logger=M.logger.name):
        for rung, timed in records:
            rpl.record(new_tokens=1024, cached_tokens=0, timed=timed, rung=rung)
            if timed:
                rpl._durations.append((0.5, 0.1, None))
        rpl.flush()
    return [r.getMessage() for r in caplog.records if "Prefill rank batch" in r.getMessage()]


def test_the_prefill_rank_batch_line_carries_the_rung_only_when_the_ladder_ran(caplog):
    base = _prefill_lines(caplog, [(None, True)])
    assert len(base) == 1 and "rung=" not in base[0]                       # unarmed: byte-equal to before
    caplog.clear()
    one = _prefill_lines(caplog, [(2, True)])
    assert one == [base[0] + " rung=2"]
    caplog.clear()
    folded = _prefill_lines(caplog, [(1, True), (1, True), (3, True)])      # K forwards fold into one line
    assert len(folded) == 1 and folded[0].endswith(" rung=1/3") and "#chunks: 3" in folded[0]
    caplog.clear()
    untimed = _prefill_lines(caplog, [(0, False), (None, False)])
    assert untimed[0].endswith("#chunks: 1 rung=0") and untimed[1].endswith("#chunks: 1")


def test_report_prefill_stats_hands_the_batch_rung_to_the_line():
    from sglang.srt.managers.scheduler_components import metrics_reporter as M

    src = inspect.getsource(M.SchedulerMetricsReporter.report_prefill_stats)
    assert 'rung=getattr(batch, "_dual_green_rung", None)' in src


# --------------------------------------------------------------------------------------- hold gate (PP0-local)
class HClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def hold_gate(**kw):
    clk = HClock()
    kw.setdefault("actuate", True)
    return G.HoldGate(clock=clk, **kw), clk


def test_hold_is_observer_by_default_and_logs_would_hold_only():
    g, clk = hold_gate(actuate=False)
    v = g.update(0.99, True, False)
    assert v == G.HoldVerdict(True, False, "observer") and g.observed_episodes == 1 and g.episodes == 0


def test_hold_hysteresis_bound_cooldown_and_the_two_releases():
    g, clk = hold_gate(hi=0.97, lo=0.90, max_s=2.0, cooldown_s=1.0)
    assert g.update(0.95, True, False).would_hold is False            # below HI
    assert g.update(0.98, True, False) == G.HoldVerdict(True, True, "hold")
    assert g.update(0.93, True, False).hold is True                    # between LO and HI: the episode goes on
    clk.t = 2.0
    v = g.update(0.99, True, False)
    assert v == G.HoldVerdict(True, False, "hold_max") and g.capped == 1   # bounded
    clk.t = 2.5
    assert g.update(0.99, True, False) == G.HoldVerdict(True, False, "cooldown")
    clk.t = 3.1
    assert g.update(0.99, True, False).hold is True and g.episodes == 2
    assert g.update(0.85, True, False) == G.HoldVerdict(False, False, "below")   # LO crossed: released
    assert g.update(0.99, True, False).hold is True
    assert g.update(0.99, False, False) == G.HoldVerdict(False, False, "d_idle")  # D empty: released AT ONCE
    assert g.holding is False


def test_hold_is_exempt_from_starvation():
    g, clk = hold_gate()
    assert g.update(0.99, True, False).hold is True
    v = g.update(0.99, True, True)
    assert v == G.HoldVerdict(True, False, "starve_exception") and g.holding is False


def test_before_forward_blocks_bounded_and_observer_never_sleeps():
    clk = HClock()
    slept = []
    g = G.HoldGate(clock=clk, actuate=True, max_s=0.1, cooldown_s=5.0)
    a = make_actuator(0, st(1, 0.75), hold=g, clock=clk)
    a._sleep = lambda s: (slept.append(s), setattr(clk, "t", clk.t + s))[-1]
    a._arena_reader = lambda sched: (0.99, "exact")
    assert a.before_forward(_launch_sched()) > 0 and slept and sum(slept) <= 0.1 + 0.05 + 1e-9
    assert a.verdict.reason in ("hold_max", "cooldown")
    slept.clear()
    g2 = G.HoldGate(clock=clk, actuate=False)
    b = make_actuator(0, st(1, 0.75), hold=g2, clock=clk)
    b._arena_reader = lambda sched: (0.99, "exact")
    b._sleep = lambda s: slept.append(s)
    assert b.before_forward(_launch_sched()) == 0.0 and not slept and b.verdict.reason == "observer"
    assert any("OBSERVER" in l and "would_hold=1" in l for l in b.logs)


def test_pick_writes_the_stage_s_real_sm_for_the_front_and_a_follower_writes_too():
    ctl = tempfile.mktemp()
    a = make_actuator(1, st(0, 1.0), pobs=G.PObsWriter(ctl, 1))
    sched = _launch_sched()
    a.apply(G.Weg2DualGreenRung(1, 3, 250000))
    a.pick(sched)
    got = G.read_pobs(ctl)
    assert got[1]["sm"] == "48" and got[1]["rung"] == "3" and got[1]["eager"] == "1"
    os.unlink(G.pobs_path(ctl, 1))


def test_pobs_files_roundtrip_to_the_front():
    ctl = tempfile.mktemp()
    w0, w1 = G.PObsWriter(ctl, 0), G.PObsWriter(ctl, 1)
    assert w0.update(sm=128, rung=1, arena_ppm=930000, would_hold=0, hold=0, seq=3)
    assert w1.update(sm=52, rung=1, seq=3)
    got = G.read_pobs(ctl)
    assert got[0]["sm"] == "128" and got[0]["arena_ppm"] == "930000" and got[1]["sm"] == "52" and 2 not in got
    assert not w0.update(sm=128, rung=1, arena_ppm=931000, would_hold=0, hold=0, seq=4)   # same key inside 1 s: no rewrite
    for r in (0, 1):
        os.unlink(G.pobs_path(ctl, r))
    assert G.read_pobs(ctl) == {}


# -------------------------------------------------------------------------- the wire through the REAL pass
def _pp_stage(rank, ctl_state, monkeypatch=None):
    a = make_actuator(rank, ctl_state)
    h = SimpleNamespace(ps=SimpleNamespace(pp_rank=rank, pp_size=3), tree_cache=None, _dual_green=a)
    h.pp_group = SimpleNamespace(is_first_rank=rank == 0, is_last_rank=rank == 2)
    h.send_req_work = None
    h.flush_wrapper = SimpleNamespace(apply_pp0_verdict=lambda *a, **k: None)
    h._weg2_store_told_armed = False
    h._weg2_vote_pass_hook = lambda reqs: None
    h._weg2_vote_after_forward = lambda reqs: reqs
    h._pp_commit_comm_work = lambda work: None
    h.sent, h.dispatched = [], []
    h._pp_send_pyobj_to_next_stage = lambda reqs, async_send=True: h.sent.append(pickle.loads(pickle.dumps(list(reqs))))
    h.process_input_requests = lambda reqs: h.dispatched.append(list(reqs))
    h.waiting_queue = []
    return h


def _nc(lst):
    return [r for r in lst if type(r).__name__ != "Weg2BurstClock"]


def _intake(h, reqs):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    SchedulerPPMixin._pp_forward_and_process_input_requests(h, reqs)


def test_rank_uniformity_one_stage_on_the_wire_every_stage_applies_the_same(monkeypatch):
    monkeypatch.setenv(S.GREEN_LADDER_ENV, "1")
    h0, h1, h2 = (_pp_stage(0, st(2, 0.5)), _pp_stage(1, st(0, 1.0)), _pp_stage(2, st(0, 1.0)))
    _intake(h0, ["req"])                                              # PP0 decides and stamps
    assert [type(r).__name__ for r in _nc(h0.sent[-1])] == ["str", "Weg2DualGreenRung"]
    assert h0.dispatched[-1] == ["req"]                               # the dispatched list stays clean
    _intake(h1, h0.sent[-1])                                          # PP1: relay verbatim, dispatch without it
    assert h1.sent[-1] == h0.sent[-1] and _nc(h1.dispatched[-1]) == ["req"]
    _intake(h2, h1.sent[-1])                                          # PP2 (last: no send)
    assert h2.sent == [] and h2.dispatched[-1] == ["req"]
    acts = [h._dual_green for h in (h0, h1, h2)]
    assert [(a.seq, a.rung, a.f) for a in acts] == [(1, 2, 0.5)] * 3   # the SAME stage, the SAME sequence
    assert acts[1].reader_cls.reads == 0 and acts[2].reader_cls.reads == 0   # followers read no ctl file
    # a change rides the next pass
    h0._dual_green.box["st"] = st(3, 0.25)
    _intake(h0, [])
    _intake(h1, h0.sent[-1])
    _intake(h2, h1.sent[-1])
    assert [(a.seq, a.rung) for a in acts] == [(2, 3)] * 3


def test_a_rank_with_the_gate_armed_but_no_actuator_still_never_dispatches_the_order(monkeypatch):
    monkeypatch.setenv(S.GREEN_LADDER_ENV, "1")
    h0, h1 = _pp_stage(0, st(1, 0.75)), _pp_stage(1, st(0, 1.0))
    h1._dual_green = None
    _intake(h0, [])
    _intake(h1, h0.sent[-1])
    assert h1.dispatched[-1] == []                                    # no Weg2DualGreenRung reaches process_input_requests


def test_off_the_gate_the_wire_is_untouched_and_the_real_pass_works_without_the_attribute(monkeypatch):
    for env in ({}, {"SGLANG_WEG2_GROUP": "P"}, dict(DUAL_P)):
        for k in list(DUAL_P) + [S.GREEN_LADDER_ENV]:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        hs = [_pp_stage(r, st(2, 0.5)) for r in range(3)]
        for h in hs:
            del h._dual_green                                         # flip / NF / INT8: the attribute never exists
        _intake(hs[0], ["req"])
        _intake(hs[1], hs[0].sent[-1])
        _intake(hs[2], hs[1].sent[-1])
        assert _nc(hs[0].sent[-1]) == ["req"] and _nc(hs[1].sent[-1]) == ["req"], env
        assert all(h.dispatched[-1] == ["req"] for h in hs), env


def test_wiring_order_in_the_pass():
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    src = inspect.getsource(SchedulerPPMixin._pp_forward_and_process_input_requests)
    stamp = src.index("_dgreen.pp0_stamp(")
    send = src.index("self._pp_send_pyobj_to_next_stage(")
    absorb = src.index("_dgreen_f.follower_absorb(")
    dispatch = src.index("self.process_input_requests(recv_reqs)")
    assert stamp < send < absorb < dispatch
    assert src.index("without_burst_clock(recv_reqs)") < src.index("_dgr.without_green_rung(_traced)") < src.index("if _traced:")


# ----------------------------------------------------------------------------------- flip / NF / INT8 unchanged
def test_launcher_argv_and_env_are_byte_equal_with_the_switch_off_and_named_with_it_on():
    from sglang.srt.weg2 import launcher as L

    base = dict(dual_layout=True, dual_priority="dynamic", dual_share_actuators="green,duty", dual_mps="on", dual_p_sm_pct=100,
                dual_p_duty=1.0, dual_d_min_rate_tps=25.0, dual_p_min_share=0.25, tag="t", dual_d_capture_prio="off",
                dual_p_mps_low_prio="off", dual_p_kv_max_tokens=4096)
    off = SimpleNamespace(**base)
    off_explicit = SimpleNamespace(**base, dual_green_ladder="off")
    assert L.dual_priority_env(off, "P") == L.dual_priority_env(off_explicit, "P")
    assert S.GREEN_LADDER_ENV not in L.dual_priority_env(off_explicit, "P") and G.HOLD_ENV not in L.dual_priority_env(off_explicit, "P")
    assert L.dual_priority_front_argv(off) == L.dual_priority_front_argv(off_explicit)
    assert "--dual-green-ladder" not in L.dual_priority_front_argv(off_explicit)
    on = SimpleNamespace(**base, dual_green_ladder="on")
    assert L.dual_priority_env(on, "P")[S.GREEN_LADDER_ENV] == "1" and G.HOLD_ENV not in L.dual_priority_env(on, "P")
    assert L.dual_priority_env(on, "D") == L.dual_priority_env(off, "D")
    assert L.dual_priority_front_argv(on)[-2:] == ["--dual-green-ladder", "on"]
    hold = SimpleNamespace(**base, dual_green_ladder="hold")
    assert L.dual_priority_env(hold, "P")[G.HOLD_ENV] == "1"
    L.refuse_dual_priority(on)
    for bad, why in (({"dual_priority": None}, "needs --dual-priority"), ({"dual_share_actuators": "chunk"}, "'green'"),
                     ({"dual_mps": "off"}, "--dual-mps on"), ({"dual_p_sm_pct": 50}, "exclude each other"),
                     ({"dual_p_kv_max_tokens": 0}, "--dual-p-kv-max-tokens")):
        ns = SimpleNamespace(**{**base, **bad, "dual_green_ladder": "on"})
        with pytest.raises(L.Weg2DualLayoutRefused) as e:
            L.refuse_dual_priority(ns)
        assert why in str(e.value), (bad, str(e.value))
    flip = SimpleNamespace(**{**base, "dual_layout": False, "dual_green_ladder": "on"})
    with pytest.raises(L.Weg2DualLayoutRefused):
        L.refuse_dual_priority(flip)
    assert L.dual_priority_env(flip, "P") == {}


def test_p_chunk_cap_without_the_ladder_keeps_the_base_warning_text():
    warns, logs = [], []
    S.maybe_chunk_cap({S.CTL_ENV: "/x", S.ACT_ENV: "green,duty"}, first_pp_rank=True, chunked_prefill_size=1024, page=64,
                      log=logs.append, warn=warns.append)
    assert any("stage 3 (green-context ladder) not built -- waits for metal probe M1" in w for w in warns)
    warns.clear()
    S.maybe_chunk_cap({S.CTL_ENV: "/x", S.ACT_ENV: "green,duty", S.GREEN_LADDER_ENV: "1"}, first_pp_rank=True,
                      chunked_prefill_size=1024, page=64, log=logs.append, warn=warns.append)
    assert not any("not built" in w for w in warns) and any("green ladder armed" in l for l in logs)


def test_the_launch_path_off_the_gate_imports_nothing_new_at_run_time():
    from sglang.srt.managers import scheduler_pp_mixin as M

    src = inspect.getsource(M.SchedulerPPMixin._pp_launch_batch)
    # every green statement sits behind the single getattr: the base launch keeps its operations
    assert src.count("_dgreen_l") >= 3 and 'getattr(self, "_dual_green", None)' in src
    assert "_dgreen_l is not None" in src and "if _gc_stream is not None" in src


# ------------------------------------------------------------------------------- review 11:03Z follow-ups
def _flap_sim(mod, bs, acc, T=120.0, dt=0.05):
    """The reviewer's closed-loop simulation with a fake clock: D rounds = solo(bs) x factor(rung); the front sees the
    EVENT COUNT of a 1.0 s window times the true tokens per round; R_min 12 tok/s (target 208 ms)."""
    clk = Clock()
    cfg = S.ShareConfig(d_min_rate_tps=12.0)
    c = mod.GreenController(cfg, "dynamic", mod.GreenConfig(), clock=clk)
    fac = (4.5, 2.8, 1.5, 1.24)
    solo = {1: 28.0, 3: 40.0, 6: 57.0}[bs]
    ev, nxt, t0, changes = [], clk.t, clk.t, 0
    for i in range(int(T / dt)):
        clk.t = t0 + i * dt
        rnd_s = solo * fac[c.rung] / 1000.0
        while nxt <= clk.t:
            ev.append(nxt)
            nxt += rnd_s
        ev = [e for e in ev if clk.t - e <= 1.0]
        rate = len(ev) * acc / 1.0 if ev else None
        d = c.observe(q_tokens=3000, b=bs, seats=4, p_rate_tps=1500.0, d_rate_tps=rate, oldest_age_s=0.0)
        changes += int(d.changed)
    return changes


def test_b2_the_flapping_case_of_the_review_is_bounded_by_the_window_and_the_latch():
    # bs6 with 3.0 true tokens per round against the assumed 2.5: the base read 55 changes in 120 s (review sim6)
    assert _flap_sim(G, 6, 3.0) <= 12
    assert _flap_sim(G, 3, 2.0) <= 6 and _flap_sim(G, 6, 2.0) <= 6 and _flap_sim(G, 1, 3.0) <= 6


def _drive(c, clk, round_ms, until, dt=0.2, cap=400):
    d = None
    for _ in range(cap):
        d = obs(c, clk, b=1, round_ms=round_ms, dt=dt)
        if until(d):
            return d
    raise AssertionError("never reached")


def test_b2_a_descent_soon_after_an_ascent_bars_that_ascent_and_the_bar_doubles():
    c, clk, d = _entered()                                           # stage 1 (75 %), target 100 ms
    d = _drive(c, clk, 50.0, lambda d: d.rung == 0)                  # calm: ascends to 100 %
    assert "d_round_calm" in d.reason and c.latches == 0
    d = _drive(c, clk, 300.0, lambda d: d.rung > 0)                  # it was wrong: over target inside 10 s
    assert c.latches == 1 and "asc_latched(0,30s)" in d.reason
    assert abs(c._asc_lock[0] - clk.t - 30.0) < 1e-6
    d = _drive(c, clk, 50.0, lambda d: "asc_locked(0" in d.reason, cap=100)    # calm again: the ascent TO 0 is barred
    assert d.rung > 0
    for _ in range(100):                                             # 20 s of calm: still barred, rung unchanged
        d = obs(c, clk, b=1, round_ms=50.0, dt=0.2)
        if clk.t > c._asc_lock[0] - 5:
            break
        assert d.rung > 0
    d = _drive(c, clk, 50.0, lambda d: d.rung == 0, cap=600)         # after the 30 s it may try again
    d = _drive(c, clk, 300.0, lambda d: d.rung > 0)                  # wrong again -> twice the bar
    assert c.latches == 2 and "asc_latched(0,60s)" in d.reason
    assert abs(c._asc_lock[0] - clk.t - 60.0) < 1e-6


def test_b2_a_descent_long_after_an_ascent_is_no_flap_and_latches_nothing():
    c, clk, d = _entered()
    _drive(c, clk, 50.0, lambda d: d.rung == 0)
    for _ in range(80):                                              # 16 s at ease at 100 %
        obs(c, clk, b=1, round_ms=60.0, dt=0.2)
    d = _drive(c, clk, 300.0, lambda d: d.rung > 0)
    assert c.latches == 0 and "asc_latched" not in d.reason


def test_b3_the_boot_probe_shape_has_at_least_eight_waves_and_the_old_one_timed_both_alike():
    def waves(shape, sm):
        tiles = (shape[0] // 128) * (shape[2] // 128)
        return -(-tiles // sm)

    for sm in (170, 128, 88, 48):
        assert waves(G.PROBE_SHAPE_DEFAULT, sm) >= 8
    assert waves((2048, 4096, 4096), 170) == waves((2048, 4096, 4096), 128) == 4     # the review's calculation
    ratio_new = waves(G.PROBE_SHAPE_DEFAULT, 128) / waves(G.PROBE_SHAPE_DEFAULT, 170)
    assert ratio_new > 1.2                                                           # 75 % rung: 16 vs 13 waves, 1.23x
    src = inspect.getsource(G.CtypesBackend.time_stream)
    assert src.index("torch.cuda.synchronize(dev)") < src.index("ctx = torch.cuda.stream")   # operands finished first
    assert src.rindex("torch.cuda.synchronize(dev)") > src.index("e1.synchronize()")


def test_b7_the_driver_struct_is_144_bytes_and_the_group_array_has_spare_slots():
    import ctypes

    assert ctypes.sizeof(G._DevResource) == G.DEV_RESOURCE_SIZEOF == 144
    assert G.SPLIT_SPARE >= 1
    assert "(_DevResource * (n + SPLIT_SPARE))" in inspect.getsource(G.CtypesBackend._split)
    assert "sizeof(_DevResource) != DEV_RESOURCE_SIZEOF" in inspect.getsource(G.CtypesBackend.__init__)


def test_b5_tp_size_above_one_is_refused_by_name_and_arms_nothing():
    env = {**DUAL_P, **LADDER_ENV}
    warns = []
    sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=0, pp_size=3, tp_size=2))
    assert G.maybe_arm(sched, env, backend=FakeBackend(170, 8), warn=warns.append, log=lambda m: None) is None
    assert len(warns) == 1 and "tp_size=2" in warns[0] and warns[0].startswith(S.FALLBACK + " mech=green")
    sched.ps.tp_size = 1
    assert G.maybe_arm(sched, env, backend=FakeBackend(170, 8), warn=lambda m: None, log=lambda m: None) is not None


def test_b4_the_vram_instrument_marker_switch_line_and_status_carry_the_allocator():
    class MemBackend(FakeBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.free = [20000 << 20, 19840 << 20]

        def mem_info(self):
            return self.free.pop(0), 32000 << 20

    logs = []
    ctl = tempfile.mktemp()
    env = {**DUAL_P, **LADDER_ENV, S.CTL_ENV: ctl}
    act = G.maybe_arm(_sched(1), env, backend=MemBackend(170, 8), log=logs.append, warn=lambda m: None)
    assert any("P vram pp=1 card_free_mib before=20000 after=19840 ladder_cost_mib=160" in l for l in logs)
    snaps = iter([{"reserved_mib": 900, "alloc_mib": 400, "retries": 0}, {"reserved_mib": 1100, "alloc_mib": 420, "retries": 2}] + [
        {"reserved_mib": 1100, "alloc_mib": 420, "retries": 2}] * 10)
    act._vram_fn = lambda: next(snaps)
    sched = _launch_sched()
    act.pick(sched)                                                  # first forward: no reading yet
    act.apply(G.Weg2DualGreenRung(1, 2, 500000))
    act.pick(sched)                                                  # stage 0 ended -> reading 900
    assert any("P rung 0->2" in l and "reserved_mib=900 alloc_retries=0" in l for l in logs)
    assert act.vram_by_rung[0]["reserved_mib"] == 900
    st_line = act.status_line()
    assert "reserved_mib=1100 alloc_mib=420 alloc_retries=2" in st_line and "vram_by_rung" in st_line and "0:900/0" in st_line
    G._reset_state_for_tests()
    assert G.vram_snapshot() is None or isinstance(G.vram_snapshot(), dict)       # no CUDA here: None, never raises
    if os.path.exists(G.pobs_path(ctl, 1)):
        os.unlink(G.pobs_path(ctl, 1))


# ---- review gaps M06 / M08 / hook behaviour / M10 / M11 / M16
def test_m06_every_green_rung_is_eager_and_rung_0_is_not():
    a = make_actuator(1, st(0, 1.0))
    sched = _launch_sched()
    a.pick(sched)
    assert G.force_eager() is False
    for rung, f, name in ((1, 0.75, "gc128"), (2, 0.5, "gc88"), (3, 0.25, "gc48")):
        a.apply(G.Weg2DualGreenRung(rung, rung, int(f * 1e6)))
        ctx, stream = a.pick(sched)
        assert stream.name == name and G.force_eager() is True, rung
    a.apply(G.Weg2DualGreenRung(9, 0, 1000000))
    a.pick(sched)
    assert G.force_eager() is False


def test_m08_the_group_rounding_is_the_rungs_sm_in_the_serve_path_not_the_wanted_count():
    be = FakeBackend(170, 8)
    a = make_actuator(1, st(0, 1.0), backend=be)
    r = a.ladder.rungs[2]
    assert r.want == 85 and r.sm == 88 and r.stream.name == "gc88"       # the driver rounded 85 up to 88
    sched = _launch_sched()
    a.pick(sched)
    a.apply(G.Weg2DualGreenRung(1, 2, 500000))
    a.pick(sched)
    assert a.active_sm == 88 and any("sm_real=88" in l for l in a.logs) and not any("sm_real=85" in l for l in a.logs)
    ctl = tempfile.mktemp()
    a.pobs = G.PObsWriter(ctl, 1)
    a.apply(G.Weg2DualGreenRung(2, 3, 250000))
    a.pick(sched)
    assert G.read_pobs(ctl)[1]["sm"] == "48"                              # 42 wanted, 48 real
    os.unlink(G.pobs_path(ctl, 1))


def test_graph_runner_hook_behaviour_green_rung_answers_eager_and_the_base_answer_is_untouched():
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner as R

    noted = []
    me = SimpleNamespace(_is_full_backend=True, _note_eager=lambda reason, fb: noted.append(reason),
                         _full_graph_ineligible_reason=lambda fb: None)
    fb = SimpleNamespace()
    G._STATE["eager"] = False
    assert R.can_run_graph(me, fb) is True and noted == []                 # unarmed / rung 0: the graph runs
    G._STATE["eager"] = True
    assert R.can_run_graph(me, fb) is False and noted == ["green_rung"]    # a green stream: eager, named
    G._STATE["eager"] = False
    assert R.can_run_graph(me, fb) is True                                 # back up to 100 %: graphs again


def test_m10_hold_from_env_actuates_only_with_the_hold_switch_and_clamps_the_knobs():
    assert G.HoldGate.from_env({}).actuate is False
    assert G.HoldGate.from_env({G.HOLD_ENV: "0"}).actuate is False
    assert G.HoldGate.from_env({S.GREEN_LADDER_ENV: "1"}).actuate is False       # the ladder switch alone never holds
    assert G.HoldGate.from_env({G.HOLD_ENV: "1"}).actuate is True
    g = G.HoldGate.from_env({G.HOLD_ENV: "1", G.HOLD_HI_ENV: "0.95", G.HOLD_LO_ENV: "0.99", G.HOLD_MAX_S_ENV: "9"})
    assert g.hi == 0.95 and g.lo < g.hi and g.max_s == 9.0                        # lo clamped under hi
    assert G.HoldGate.from_env({G.HOLD_HI_ENV: "junk"}).hi == G.HOLD_ARENA_HI_DEFAULT


def test_m11_the_trace_filter_hides_the_stage_order_only_behind_the_switch(monkeypatch):
    monkeypatch.setenv(S.GREEN_LADDER_ENV, "1")
    h0, h1 = _pp_stage(0, st(1, 0.75)), _pp_stage(1, st(0, 1.0))
    _intake(h0, [])                                                         # PP0 stamps; PP0's own list is empty
    assert getattr(h0, "_pp_req_trace_n", 0) == 0
    _intake(h1, h0.sent[-1])                                                # the follower's list = [stamp] only
    assert getattr(h1, "_pp_req_trace_n", 0) == 0                           # not traced as a request
    _intake(h1, ["req"] + h0.sent[-1])
    assert h1._pp_req_trace_n == 1                                          # a real request still is
    monkeypatch.delenv(S.GREEN_LADDER_ENV)
    h2 = _pp_stage(1, st(0, 1.0))
    del h2._dual_green
    _intake(h2, ["req"])
    assert h2._pp_req_trace_n == 1


def test_m16_the_stamp_key_is_rung_and_the_fraction_rounded_to_four_digits():
    t = [0.0]
    a0 = make_actuator(0, st(2, 0.5), clock=lambda: t[0])
    assert a0.pp0_stamp([])[1] is not None
    a0.box["st"] = st(2, 0.50004)                                           # same key at 4 digits: no restamp
    t[0] = 0.1
    assert a0.pp0_stamp([])[1] is None
    a0.box["st"] = st(2, 0.5002)                                            # another fraction at 4 digits: restamp
    t[0] = 0.2
    c = a0.pp0_stamp([])[1]
    assert c is not None and c.f_ppm == 500200
    a0.box["st"] = st(3, 0.5002)                                            # another rung, same fraction: restamp
    t[0] = 0.3
    assert a0.pp0_stamp([])[1].rung == 3


# ----------------------------------------------------------------------------------------------- the mutants
_SRC = inspect.getsource(G)


def _mutant(*pairs):
    """dual_green with its source mutated -- a module object of its own, sys.modules untouched afterwards."""
    src = _SRC
    for old, new in pairs:
        assert src.count(old) == 1, ("mutation target not unique/found", old)
        src = src.replace(old, new)
    name = "dual_green_mutant_%d" % abs(hash(pairs))
    mod = types.ModuleType(name)
    mod.__file__ = "<mutant>"
    sys.modules[name] = mod
    try:
        exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def _gate_check(mod):
    """True when every wrong gate leaves the ladder unarmed and the right one arms it."""
    wrong = [{**LADDER_ENV}, {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", **LADDER_ENV},
             {"SGLANG_WEG2_DUAL_LAYOUT": "0", "SGLANG_WEG2_GROUP": "P", **LADDER_ENV}]
    return (not any(mod.armed(e) for e in wrong)) and mod.armed({**DUAL_P, **LADDER_ENV})


def _dwell_check(mod):
    """No descent 0.2 s after a change (inside desc_min_s), one after 1.1 s."""
    clk = Clock()
    c = mod.GreenController(S.ShareConfig(d_min_rate_tps=25.0), "dynamic", clock=clk)

    def ob(dt, round_ms):
        clk.t += dt
        return c.observe(q_tokens=0.0, b=1, seats=6, p_rate_tps=None,
                         d_rate_tps=None if round_ms is None else 2500.0 / round_ms, oldest_age_s=0.0)

    clk.t += 1.0
    ob(0.0, None)
    early = ob(0.2, 150.0).rung
    late = ob(0.9, 150.0).rung
    return early == 1 and late == 2


def _mapping_check(mod):
    ld = mod.GreenLadder(FakeBackend(170, 8), FRACTIONS, log=lambda m: None, warn=lambda m: None).build()
    got = [ld.entry(f) for f in (0.5, 0.75, 0.25)]
    return all(e is not None for e in got) and [e.sm for e in got] == [88, 128, 48]


def _bs_direction_check(mod):
    out = []
    for bs in (1, 3, 6):
        clk = Clock()
        c = mod.GreenController(S.ShareConfig(), "dynamic", clock=clk)
        clk.t += 1.0
        out.append(c.observe(q_tokens=0.0, b=bs, seats=6, p_rate_tps=None, d_rate_tps=None).rung)
    return out == [1, 2, 3]                                          # the more D bs, the deeper (table, no target)


def _wire_check(mod):
    """Three stages through the wire: every stage ends with the SAME stage."""
    ld = [mod.GreenLadder(FakeBackend(170, 8), FRACTIONS, log=lambda m: None, warn=lambda m: None).build() for _ in range(3)]

    class R:
        def read(self):
            return st(2, 0.5)

    acts = [mod.GreenActuator(ld[i], R(), pp_rank=i, pp_size=3, log=lambda m: None) for i in range(3)]
    wire, cmd = acts[0].pp0_stamp([])
    w = pickle.loads(pickle.dumps(wire)) if False else wire
    for a in acts[1:]:
        w = a.follower_absorb(list(wire))
    return all(a.rung == 2 and a.seq == 1 for a in acts)


def _idle_check(mod):
    clk = Clock()
    c = mod.GreenController(S.ShareConfig(d_min_rate_tps=25.0), "dynamic", clock=clk)
    clk.t += 1.0
    c.observe(q_tokens=0.0, b=6, seats=6, p_rate_tps=None, d_rate_tps=None)
    clk.t += 0.01
    return c.observe(q_tokens=0.0, b=0, seats=6, p_rate_tps=None, d_rate_tps=None).rung == 0


def _starve_hold_check(mod):
    clk = HClock()
    g = mod.HoldGate(clock=clk, actuate=True)
    g.update(0.99, True, False)
    return g.update(0.99, True, True).hold is False


def _bound_check(mod):
    clk = HClock()
    g = mod.HoldGate(clock=clk, actuate=True, max_s=2.0)
    g.update(0.99, True, False)
    clk.t = 3.0
    return g.update(0.99, True, False).hold is False


def _eager_check(mod):
    mod._STATE["eager"] = True
    try:
        return mod.force_eager() is True
    finally:
        mod._STATE["eager"] = False


def _fallback_check(mod):
    """A rung whose create fails must not be served and must be named (never silently the full stream)."""
    be = FakeBackend(170, 8, fail_create=(85,))
    warns = []
    ld = mod.GreenLadder(be, FRACTIONS, probe=False, log=lambda m: None, warn=warns.append).build()   # probe off: the
    return (not ld.serves(0.5)) and ld.serves(0.75) and any("rung 2" in w for w in warns)        # bookkeeping alone


def _sm_check(mod):
    """The serve path reports the DRIVER's rounded SM count (85 wanted -> 88), never the wanted one."""
    ld = mod.GreenLadder(FakeBackend(170, 8), FRACTIONS, probe=False, log=lambda m: None, warn=lambda m: None).build()

    class R:
        def read(self):
            return st(2, 0.5)

    a = mod.GreenActuator(ld, R(), pp_rank=1, pp_size=3, log=lambda m: None, vram_fn=lambda: None)
    a.apply(mod.Weg2DualGreenRung(1, 2, 500000))
    a.pick(_launch_sched())
    return a.active_sm == 88


def _stamp_key_check(mod):
    ld = mod.GreenLadder(FakeBackend(170, 8), FRACTIONS, probe=False, log=lambda m: None, warn=lambda m: None).build()
    box = {"st": st(2, 0.5)}

    class R:
        def read(self):
            return box["st"]

    t = [0.0]
    a = mod.GreenActuator(ld, R(), pp_rank=0, pp_size=3, log=lambda m: None, clock=lambda: t[0], vram_fn=lambda: None)
    a.pp0_stamp([])
    box["st"] = st(2, 0.5002)
    t[0] = 0.2
    return a.pp0_stamp([])[1] is not None


def _latch_check(mod):
    return _flap_sim(mod, 6, 3.0) <= 12


MUTANTS = [
    ("gate out", (("    if not _pk.armed(e):\n        return False\n", "    if False:\n        return False\n"),), _gate_check),
    ("dwell out", (("desc_min_s: float = 1.0 ", "desc_min_s: float = 0.0 "),), _dwell_check),
    ("stage mapping wrong (farthest fraction)", (("min(range(len(self.fractions)), key=lambda k: abs(", "max(range(len(self.fractions)), key=lambda k: abs("),), _mapping_check),
    ("bs direction reversed", (("((2, 1, 0), (4, 2, 1), (10 ** 9, 3, 2))", "((2, 3, 2), (4, 2, 1), (10 ** 9, 1, 0))"),), _bs_direction_check),
    ("wire only PP0 (follower ignores the stamp)", (("        for c in sorted(cmds, key=lambda c: c.seq):\n            self.apply(c)\n", "        for c in sorted(cmds, key=lambda c: c.seq):\n            pass\n"),), _wire_check),
    ("D-empty jump out", (('target, reason, immediate = 0, "d_idle", True', 'target, reason, immediate = 0, "d_idle", False'),), _idle_check),
    ("hold starvation exemption out", (("        if starve:                                         # the starvation clamp wins over a hold\n", "        if False:\n"),), _starve_hold_check),
    ("hold bound out", (("            if now - self._hold_t0 >= self.max_s:\n", "            if False:\n"),), _bound_check),
    ("force_eager always False", (('    return bool(_STATE["eager"])', "    return False"),), _eager_check),
    ("latch never bars", (("                                self._asc_lock[to] = now + lock_s\n", "                                pass\n"),), _latch_check),
    ("serve path reports the wanted SM", (("entry.sm if entry is not None else (self.ladder.info.sm_total", "entry.want if entry is not None else (self.ladder.info.sm_total"),), _sm_check),
    ("stamp key rung only", (("            key = (int(st.rung), round(float(st.fraction), 4))", "            key = (int(st.rung),)"),), _stamp_key_check),
    ("failed rung served as if healthy", (("                self.failed[i] = f\"{type(e).__name__}: {e}\"\n                self.counters[\"create_failed\"] += 1\n", "                self.failed[i] = f\"{type(e).__name__}: {e}\"\n                self.counters[\"create_failed\"] += 1\n                self.rungs[i] = Rung(index=i, fraction=f, want=want, sm=0, stream=None)\n"),), _fallback_check),
]


@pytest.mark.parametrize("name,pairs,check", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_each_mutant_turns_its_check_red_and_the_real_module_is_green(name, pairs, check):
    assert check(G) is True, f"the real module fails its own check ({name})"
    assert check(_mutant(*pairs)) is False, f"mutant '{name}' survived"
