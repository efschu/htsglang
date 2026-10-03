# SPDX-License-Identifier: Apache-2.0
"""Item 800 DUAL-SHARE: P vs D precedence selectable and dynamic in the dual layout.

User order 03.10. ~18:55-19:15Z: many waiting prefill tokens + D bs1 -> P gets more;
few waiting + a large D bs -> D gets more; hardware-generic; a missing mechanism is a
named W-DUAL-SHARE-FALLBACK line, never silent; default off = byte-identical.

DANGER DIRECTIONS guarded here:
* the characteristic: tau x b matrix on the rung ladder, static modes, D idle = P full;
* the hysteresis: dwell 0.5 s toward D / 2 s toward P, one rung per decision, dead band,
  flaps counted not refused; immediate only for D idle, starvation, operator mode change;
* guards: D minimum rate, starvation clamp, P minimum share;
* the ctl file: atomic rewrite, a stale/missing/malformed file is P FULL (named once);
* actuators: PP0-only chunk cap (page floor, graph-bucket snap, never widens), duty;
* stage 1: D capture priority from the device's runtime range, MPS client priority
  only with MPS and a new-enough driver -- otherwise named;
* stage 3 prep: the SM ladder from simulated SM counts / granularities, nothing hard-wired;
* launcher: every switch default off -> env/argv byte-identical; refused outside dual.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import re
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import dual_share as S  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _ctrl(mode="dynamic", **kw):
    clk = Clock(100.0)
    cfg = S.ShareConfig(**kw)
    return S.ShareController(cfg, mode, clock=clk), clk


def _settle(c, clk, steps=40, dt=0.25, **obs):
    d = None
    for _ in range(steps):
        clk.t += dt
        d = c.observe(**obs)
    return d


# ------------------------------------------------------------------ config

class TestConfig:
    def test_defaults_are_the_research_ladder(self):
        cfg = S.ShareConfig()
        assert cfg.rungs == (1.0, 0.75, 0.5, 0.25)
        assert cfg.max_rung() == 3
        assert cfg.static_rung("balanced") == 2 and cfg.static_rung("d") == 3

    def test_env_overrides(self):
        env = {"SGLANG_WEG2_DUAL_SHARE_RUNGS": "1,0.5", "SGLANG_WEG2_DUAL_SHARE_TAU_EDGES_S": "1,5",
               "SGLANG_WEG2_DUAL_SHARE_MATRIX": "0,1,1,1;0,0,1,1;0,0,0,1",
               "SGLANG_WEG2_DUAL_SHARE_STATIC": "balanced=1,d=1",
               "SGLANG_WEG2_DUAL_SHARE_DWELL_TO_P_S": "3"}
        cfg = S.config_from_env(env, d_min_rate_tps=20, p_min_share=0.5)
        assert cfg.rungs == (1.0, 0.5) and cfg.tau_edges_s == (1.0, 5.0)
        assert cfg.dwell_to_p_s == 3.0 and cfg.d_min_rate_tps == 20.0 and cfg.max_rung() == 1

    @pytest.mark.parametrize("kw", [{"rungs": (0.9, 0.5)}, {"rungs": (1.0, 0.5, 0.6)},
                                    {"tau_edges_s": (5.0, 2.0)}, {"b_edges": (0.5, 1.5)},
                                    {"matrix": ((0, 1, 2, 9), (0, 1, 2, 2), (0, 0, 1, 2))},
                                    {"p_min_share": 0.0}, {"static": (("x", 1),)}])
    def test_bad_config_is_refused(self, kw):
        with pytest.raises(ValueError):
            S.ShareConfig(**kw)

    def test_actuators_parse(self):
        assert S.parse_actuators("chunk, duty,chunk") == ("chunk", "duty")
        with pytest.raises(ValueError):
            S.parse_actuators("chunk,mps")
        with pytest.raises(ValueError):
            S.parse_actuators("")


# ------------------------------------------------------------------ the characteristic

class TestCharacteristic:
    def test_many_waiting_and_d_bs1_gives_p_everything(self):
        # tau = 120000 / 2400 = 50 s (high), B = 1 of 6 seats (low): rung 0
        c, clk = _ctrl()
        d = _settle(c, clk, q_tokens=120000, b=1, seats=6, p_rate_tps=2400, d_rate_tps=None)
        assert d.rung == 0 and d.fraction == 1.0 and "tau=high" in d.reason

    def test_few_waiting_and_large_d_bs_gives_d_the_most(self):
        c, clk = _ctrl()
        d = _settle(c, clk, q_tokens=500, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        assert d.rung == 3 and d.fraction == 0.25 and "tau=low" in d.reason and "b=high" in d.reason

    @pytest.mark.parametrize("q,b,want", [(120000, 3, 1), (120000, 6, 2), (12000, 1, 1),
                                          (12000, 3, 2), (12000, 6, 2), (500, 1, 1), (500, 3, 2)])
    def test_matrix_cells(self, q, b, want):
        c, clk = _ctrl()
        d = _settle(c, clk, q_tokens=q, b=b, seats=6, p_rate_tps=2400, d_rate_tps=None)
        assert d.rung == want, d

    def test_d_idle_is_p_full_at_once_in_every_mode(self):
        for mode in S.MODES:
            c, clk = _ctrl(mode)
            _settle(c, clk, q_tokens=10, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
            clk.t += 0.01  # far inside every dwell
            d = c.observe(q_tokens=10, b=0, seats=6, p_rate_tps=2400, d_rate_tps=None)
            assert d.rung == 0 and d.reason.startswith(("d_idle", "mode_p")), (mode, d)

    def test_static_modes(self):
        for mode, want in (("p", 0), ("balanced", 2), ("d", 3)):
            c, clk = _ctrl(mode)
            d = _settle(c, clk, q_tokens=10 ** 6, b=1, seats=6, p_rate_tps=2400, d_rate_tps=None)
            assert d.rung == want, (mode, d)

    def test_unknown_p_rate_takes_the_mid_row_named(self):
        c, clk = _ctrl()
        d = _settle(c, clk, q_tokens=500, b=6, seats=6, p_rate_tps=None, d_rate_tps=None)
        assert d.tau_s is None and "tau=unknown" in d.reason and d.rung == 2

    def test_deadband_holds_the_class_near_an_edge(self):
        assert S._classify(10.5, (2.0, 10.0), 1, 0.2) == 1      # 10.5 < 12: stays mid
        assert S._classify(12.5, (2.0, 10.0), 1, 0.2) == 2
        assert S._classify(1.8, (2.0, 10.0), 1, 0.2) == 1       # 1.8 > 1.6: stays mid
        assert S._classify(1.5, (2.0, 10.0), 1, 0.2) == 0
        assert S._classify(50.0, (2.0, 10.0), None, 0.2) == 2


# ------------------------------------------------------------------ hysteresis

class TestHysteresis:
    def test_one_rung_per_decision_and_dwell_toward_d(self):
        c, clk = _ctrl()
        obs = dict(q_tokens=500, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        clk.t += 0.25
        d = c.observe(**obs)
        assert d.rung == 1 and d.changed        # first step: no previous change
        clk.t += 0.25
        d = c.observe(**obs)
        assert d.rung == 1 and d.held and "dwell_hold" in d.reason  # 0.25 < 0.5
        clk.t += 0.25
        d = c.observe(**obs)
        assert d.rung == 2                      # 0.5 s after the last change
        clk.t += 0.5
        assert c.observe(**obs).rung == 3

    def test_toward_p_waits_two_seconds(self):
        c, clk = _ctrl()
        _settle(c, clk, q_tokens=500, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        assert c.rung == 3
        obs = dict(q_tokens=10 ** 6, b=1, seats=6, p_rate_tps=2400, d_rate_tps=None)
        ts = []
        for _ in range(60):
            clk.t += 0.25
            d = c.observe(**obs)
            if d.changed:
                ts.append((clk.t, d.rung))
        assert [r for _, r in ts] == [2, 1, 0]
        gaps = [b[0] - a[0] for a, b in zip(ts, ts[1:])]
        assert all(g >= 2.0 for g in gaps), gaps

    def test_flaps_are_counted_not_refused(self):
        c, clk = _ctrl(dwell_to_d_s=0.0, dwell_to_p_s=0.0, ewma_s=0.0, deadband=0.0)
        hi = dict(q_tokens=10 ** 6, b=1, seats=6, p_rate_tps=2400, d_rate_tps=None)
        lo = dict(q_tokens=10, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        for obs in (lo, hi, lo, hi):
            clk.t += 0.25
            d = c.observe(**obs)
        assert d.flaps >= 2
        line = S.decision_line(d, ("chunk",))
        assert re.search(r"^DUAL-SHARE mode=dynamic tau=\S+ dbs=1/6 .* rung=\d->\d .* actuator=chunk "
                         r"reason=\S+ flaps=\d+ dwell_s=", line), line

    def test_operator_mode_change_jumps(self):
        c, clk = _ctrl("p")
        _settle(c, clk, q_tokens=10, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        assert c.rung == 0
        assert c.set_mode("d") is True
        clk.t += 0.01
        assert c.observe(q_tokens=10, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None).rung == 3
        with pytest.raises(ValueError):
            c.set_mode("x")

    def test_held_target_logs_once(self):
        c, clk = _ctrl()
        obs = dict(q_tokens=500, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        clk.t += 0.25
        assert c.should_log(c.observe(**obs))           # change
        clk.t += 0.1
        assert c.should_log(c.observe(**obs))           # first hold of (1, 3)
        clk.t += 0.1
        assert not c.should_log(c.observe(**obs))       # same hold again


# ------------------------------------------------------------------ guards

class TestGuards:
    def test_min_rate_steps_toward_d_and_relaxes(self):
        c, clk = _ctrl("balanced", d_min_rate_tps=20.0, ewma_s=0.0)
        slow = dict(q_tokens=10 ** 6, b=1, seats=6, p_rate_tps=2400, d_rate_tps=10.0)
        d = _settle(c, clk, steps=20, **slow)
        assert d.rung == 3 and "rate_low" in d.reason
        fast = dict(slow, d_rate_tps=40.0)
        d = _settle(c, clk, steps=200, **fast)
        assert d.rung == 2  # back to the static rung, never above it

    def test_min_rate_off_in_mode_p(self):
        c, clk = _ctrl("p", d_min_rate_tps=20.0, ewma_s=0.0)
        d = _settle(c, clk, q_tokens=1, b=6, seats=6, p_rate_tps=2400, d_rate_tps=1.0)
        assert d.rung == 0

    def test_starvation_clamps_at_once(self):
        c, clk = _ctrl("d")
        _settle(c, clk, q_tokens=10, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        assert c.rung == 3
        clk.t += 0.01
        d = c.observe(q_tokens=10, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None, oldest_age_s=61.0)
        assert d.rung == 1 and "starve" in d.reason

    def test_p_min_share_floor(self):
        c, clk = _ctrl("d", p_min_share=0.5)
        d = _settle(c, clk, q_tokens=10, b=6, seats=6, p_rate_tps=2400, d_rate_tps=None)
        assert d.rung == 2 and d.fraction == 0.5 and "p_min_share" in d.reason


# ------------------------------------------------------------------ meters

class TestMeters:
    def test_p_rate_prefers_d_idle_samples(self):
        m = S.PRateMeter()
        assert m.rate() == (None, "none")
        m.note(1000, 1.0, d_idle=False)
        r, src = m.rate()
        assert r == 1000 and src.startswith("p90_all")
        for _ in range(3):
            m.note(3000, 1.0, d_idle=True)
        r, src = m.rate()
        assert r == 3000 and src.startswith("idle_median")
        m.note(0, 1.0, True)
        m.note(10, None, True)  # ignored

    def test_d_rate_calibrates_tokens_per_event(self):
        clk = Clock(0.0)
        m = S.DRateMeter(clock=clk)
        for _ in range(20):
            clk.t += 0.1
            m.chunk("a", 1)
        m.done("a", 60)                          # 20 events -> 60 tokens
        assert m.tok_per_event > 1.0
        assert m.rate() == pytest.approx(60 / 1.9)  # completed median while none is live
        for _ in range(30):
            clk.t += 0.1
            m.chunk("b", 1)
        assert m.rate() == pytest.approx(10 * m.tok_per_event, rel=0.1)
        m.prune({})
        assert "b" not in m._live

    def test_sse_events(self):
        assert S.sse_events(b"data: a\n\ndata: b\n\n") == 2
        assert S.sse_events(b"{}") == 1 and S.sse_events(b"") == 0


# ------------------------------------------------------------------ ctl file

class TestCtl:
    def _dec(self, rung=2, b=3):
        return S.Decision(mode="dynamic", rung=rung, prev_rung=0, target=rung, fraction=(1, .75, .5, .25)[rung],
                          tau_s=1.5, b=b, seats=6, q_tokens=100, p_rate_tps=2000, d_rate_tps=25.0, reason="x",
                          changed=True, held=False, flaps=0, dwell_s=0.0)

    def test_roundtrip_and_rewrite_only_on_change_or_heartbeat(self, tmp_path):
        path = str(tmp_path / "busy.ctl")
        clk = Clock(0.0)
        w = S.CtlWriter(path, heartbeat_s=1.0, clock=clk)
        assert w.update(self._dec())
        assert not w.update(self._dec())
        clk.t += 1.1
        assert w.update(self._dec())                       # heartbeat
        assert w.update(self._dec(rung=3))                 # change
        r = S.CtlReader(path, log=lambda s: None)
        st = r.read()
        assert st.ok and st.rung == 3 and st.fraction == 0.25 and st.d_busy and st.mode == "dynamic"

    def test_missing_stale_malformed_are_p_full_named_once(self, tmp_path):
        lines = []
        path = str(tmp_path / "busy.ctl")
        clk = Clock(0.0)
        r = S.CtlReader(path, clock=clk, log=lines.append)
        st = r.read()
        assert not st.ok and st.fraction == 1.0 and st.rung == 0
        clk.t += 1.0
        r.read()
        assert len(lines) == 1 and lines[0].startswith("W-DUAL-SHARE-FALLBACK mech=ctl card=all reason=ctl unreadable")
        S.CtlWriter(path).update(self._dec())
        clk.t += 1.0
        assert r.read().ok
        os.utime(path, (time.time() - 60, time.time() - 60))
        clk.t += 1.0
        st = r.read()
        assert not st.ok and st.fraction == 1.0 and "stale" in lines[-1]
        with open(path, "w") as f:
            f.write("garbage\n")
        clk.t += 1.0
        assert r.read().fraction == 1.0


# ------------------------------------------------------------------ actuators

class _FixedReader:
    def __init__(self, rung, f, busy=True):
        self.path = "/dev/null"
        self.st = S.CtlState(rung=rung, fraction=f, d_busy=busy, mode="d", seq=1, ok=True)

    def read(self):
        return self.st


class TestChunkCap:
    def test_width_for_floors_to_page_and_snaps_to_a_bucket(self):
        assert S.chunk_width_for(1.0, 1024, 64) == (1024, "full")
        assert S.chunk_width_for(0.5, 1024, 64, (512,)) == (512, "graph")
        assert S.chunk_width_for(0.75, 1024, 64, (512,)) == (512, "graph")   # 768: bucket 512 in (384, 768]
        assert S.chunk_width_for(0.25, 1024, 64, (512,)) == (256, "eager")   # no bucket in (128, 256]
        assert S.chunk_width_for(0.3, 1000, 64) == (256, "eager")            # 300 -> page floor 256
        assert S.chunk_width_for(0.01, 1024, 64) == (64, "eager")            # never below one page

    def test_cap_never_widens(self):
        lines = []
        cap = S.ChunkCap(_FixedReader(3, 0.25), 1024, 64, (512,), log=lines.append)
        assert cap.cap(1024) == 256 and cap.cap(128) == 128
        assert cap.capped == 1 and "rung=3" in lines[0]
        assert S.ChunkCap(_FixedReader(0, 1.0), 1024, 64).cap(1024) == 1024

    def test_maybe_chunk_cap_gates(self, tmp_path):
        env = {S.CTL_ENV: str(tmp_path / "c.ctl"), S.ACT_ENV: "chunk", S.BUCKETS_ENV: "512"}
        warns = []
        kw = dict(chunked_prefill_size=1024, page=64, log=lambda s: None, warn=warns.append)
        assert S.maybe_chunk_cap({}, first_pp_rank=True, **kw) is None
        assert S.maybe_chunk_cap(env, first_pp_rank=False, **kw) is None        # PP0 decides alone
        assert S.maybe_chunk_cap(dict(env, **{S.ACT_ENV: "duty"}), first_pp_rank=True, **kw) is None
        cap = S.maybe_chunk_cap(env, first_pp_rank=True, **kw)
        assert cap is not None and cap.buckets == (512,) and warns == []
        assert S.maybe_chunk_cap(env, first_pp_rank=True, **dict(kw, chunked_prefill_size=None)) is None
        assert warns[-1].startswith("W-DUAL-SHARE-FALLBACK mech=chunk card=all reason=no --chunked-prefill-size")

    def test_green_is_a_named_fallback_to_chunk(self, tmp_path):
        warns = []
        env = {S.CTL_ENV: str(tmp_path / "c.ctl"), S.ACT_ENV: "green"}
        cap = S.maybe_chunk_cap(env, first_pp_rank=True, chunked_prefill_size=1024, page=64,
                                log=lambda s: None, warn=warns.append)
        assert cap is not None
        assert warns[0].startswith("W-DUAL-SHARE-FALLBACK mech=green card=all reason=stage 3")

    def test_apply_chunk_cap_identity_without_cap(self):
        assert S.apply_chunk_cap(SimpleNamespace(), 1024) == 1024
        sched = SimpleNamespace(_dual_share_chunk=S.ChunkCap(_FixedReader(2, 0.5), 1024, 64, log=lambda s: None))
        assert S.apply_chunk_cap(sched, 1024) == 512


class TestShareDuty:
    def test_pause_follows_the_rung(self):
        slept = []
        d = S.ShareDuty(_FixedReader(2, 0.5), sleep=slept.append)
        assert d.before_forward() == 0.0          # no forward measured yet
        d.after_forward(0.1)
        assert d.before_forward() == pytest.approx(0.1)
        d.after_forward(10.0)
        assert d.before_forward() == S.MAX_SLEEP_S
        assert S.ShareDuty(_FixedReader(0, 1.0), sleep=slept.append).pause_s() == 0.0
        idle = S.ShareDuty(_FixedReader(3, 0.25, busy=False))
        idle.after_forward(0.1)
        assert idle.pause_s() == 0.0

    def test_from_env(self):
        assert S.ShareDuty.from_env({}) is None
        assert S.ShareDuty.from_env({S.DUTY_ENV: "1"}) is None
        assert S.ShareDuty.from_env({S.DUTY_ENV: "1", S.CTL_ENV: "/x"}) is not None


# ------------------------------------------------------------------ stage 1

class TestStage1:
    def test_d_capture_gate(self):
        assert not S.d_capture_armed({})
        assert not S.d_capture_armed({S.D_CAPTURE_PRIO_ENV: "1", "SGLANG_WEG2_GROUP": "P"})
        assert not S.d_capture_armed({"SGLANG_WEG2_GROUP": "D"})
        assert S.d_capture_armed({S.D_CAPTURE_PRIO_ENV: "1", "SGLANG_WEG2_GROUP": "D"})
        assert S.d_capture_stream({}) is None  # never touches torch when off

    def test_capture_priority_from_the_runtime_range(self):
        assert S.pick_capture_priority((0, -5))[0] == -5
        assert S.pick_capture_priority((0, -1))[0] == -1
        p, why = S.pick_capture_priority((0, 0))
        assert p is None and "one stream priority level" in why
        assert S.pick_capture_priority(None)[0] is None

    def test_mps_client_priority(self):
        env, line = S.mps_client_priority_env(mps_on=False, driver_text=None)
        assert env == {} and line.startswith("W-DUAL-SHARE-FALLBACK mech=mps_client_priority")
        old = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  525.60.13  Wed Nov 30 2022"
        env, line = S.mps_client_priority_env(mps_on=True, driver_text=old)
        assert env == {} and "driver 525 < 535" in line
        new = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.58.03  Release Build"
        env, line = S.mps_client_priority_env(mps_on=True, driver_text=new)
        assert env == {"CUDA_MPS_CLIENT_PRIORITY": "1"} and "driver 595" in line
        env, line = S.mps_client_priority_env(mps_on=True, driver_text=None)
        assert env == {"CUDA_MPS_CLIENT_PRIORITY": "1"} and "unverified" in line


# ------------------------------------------------------------------ hardware-generic (stage 3 prep)

class TestHardwareGeneric:
    @pytest.mark.parametrize("sms,gran,minp,want", [
        (68, 2, 2, [68, 50, 34, 16]),      # sm86-like simulated
        (170, 8, 8, [170, 120, 80, 40]),   # sm120-like simulated
        (132, 8, 8, [132, 96, 64, 32]),    # a third simulated card
        (46, 2, 4, [46, 34, 22, 10]),
    ])
    def test_ladder_rounds_down_to_the_read_granularity(self, sms, gran, minp, want):
        lad, why = S.sm_ladder(sms, (1.0, 0.75, 0.5, 0.25), gran, minp)
        assert [n for _, n in lad] == want and why is None
        assert all(n % gran == 0 for _, n in lad[1:])

    def test_ladder_fallbacks_are_named(self):
        lad, why = S.sm_ladder(10, (1.0, 0.75, 0.5, 0.25), 8, 8)
        assert [n for _, n in lad] == [10] and "1 distinct rung" in why
        assert S.sm_ladder(80, (1.0, 0.5), None)[1].startswith("granularity not read")
        assert S.sm_ladder(0, (1.0,), 2)[1] == "SM count unreadable"
        lad, _ = S.sm_ladder(16, (1.0, 0.75, 0.5, 0.25), 8, 8)
        assert [n for _, n in lad] == [16, 8]   # 12 -> 8 and 4 -> dropped / duplicate

    def test_no_card_is_hard_wired(self):
        """CODE only (comments/docstrings may cite measurements): no SM count,
        card name or arch literal as a number, name or short string."""
        import io
        import tokenize

        bad = {"68", "170", "84", "82", "3080", "5090", "sm86", "sm120", "sm_86", "sm_120"}
        hits = []
        for tok in tokenize.generate_tokens(io.StringIO(inspect.getsource(S)).readline):
            if tok.type in (tokenize.NUMBER, tokenize.NAME) and tok.string in bad:
                hits.append(tok)
            elif tok.type == tokenize.STRING and not tok.string.startswith(('"' * 3, "'" * 3)):
                hits += [tok for b in bad if re.search(r"(?<![\w.])%s(?![\w.])" % b, tok.string)]
        assert hits == [], hits


# ------------------------------------------------------------------ front glue

class TestFrontShare:
    def test_tick_reads_queue_plus_p_inflight_and_writes_ctl(self, tmp_path):
        lines = []
        clk = Clock(0.0)
        path = str(tmp_path / "x.ctl")
        fs = S.FrontShare(ctl=path, mode="dynamic", actuators=("chunk",), cfg=S.ShareConfig(), clock=clk,
                          log=lines.append)
        for _ in range(3):
            fs.note_leg1_done("r0", 2400, 1.0, d_idle=True)
        fs.note_leg1_start("r1", 100)
        q = [SimpleNamespace(est_uncached=200, t_arrive=1000.0)]
        for _ in range(12):
            clk.t += 0.25
            d = fs.tick(queue=q, p_outstanding={"r1": 1000.0}, d_outstanding={"a": 1, "b": 1, "c": 1},
                        seats=4, d_handoff=1, now_wall=1001.0)
        assert d.b == 4 and d.q_tokens == pytest.approx(300, rel=0.01) and d.tau_s < 1.0
        assert d.rung == 3
        st = S.CtlReader(path, log=lambda s: None).read()
        assert st.ok and st.rung == 3
        assert any(ln.startswith("DUAL-SHARE mode=dynamic") for ln in lines)
        fs.tick(queue=[], p_outstanding={}, d_outstanding={}, seats=4, now_wall=1002.0)
        assert fs.p_rest == {}
        snap = fs.set_mode("p", p_min_share=0.5)
        assert snap["mode"] == "p" and snap["p_min_share"] == 0.5

    def test_from_args_off(self):
        assert S.FrontShare.from_args(ctl="", mode="dynamic", actuators="chunk", d_min_rate_tps=0,
                                      p_min_share=0.25) is None
        assert S.FrontShare.from_args(ctl="/x", mode="", actuators="chunk", d_min_rate_tps=0,
                                      p_min_share=0.25) is None

    def test_admin_endpoint(self, tmp_path):
        from sglang.srt.weg2 import front as F

        fs = S.FrontShare(ctl=str(tmp_path / "c.ctl"), mode="dynamic", actuators=("chunk",),
                          cfg=S.ShareConfig(), log=lambda s: None)

        def req(method, body=None, remote="127.0.0.1", headers=None):
            async def js():
                return body

            return SimpleNamespace(method=method, remote=remote, headers=headers or {}, json=js)

        def run(front, r):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(F.Front.handle_dual_priority(front, r))
            finally:
                loop.close()

        front = SimpleNamespace(_dual_share=fs, admin_key=None)
        assert run(front, req("POST", {"mode": "d"})).status == 200 and fs.ctrl.mode == "d"
        assert run(front, req("POST", {"mode": "x"})).status == 400
        assert run(front, req("GET", remote="10.0.0.5")).status == 403
        keyed = SimpleNamespace(_dual_share=fs, admin_key="k")
        assert run(keyed, req("POST", {"mode": "p"}, remote="10.0.0.5")).status == 401
        ok = run(keyed, req("POST", {"mode": "p"}, remote="10.0.0.5", headers={"Authorization": "Bearer k"}))
        assert ok.status == 200 and fs.ctrl.mode == "p"
        assert run(SimpleNamespace(_dual_share=None, admin_key=None), req("GET")).status == 404


# ------------------------------------------------------------------ launcher

class TestLauncher:
    @staticmethod
    def _ns(*extra):
        from sglang.srt.weg2 import launcher as L

        return L.build_parser().parse_args(["--tree", "/x", "--tag", "t", *extra])

    def test_default_off_is_byte_identical(self):
        from sglang.srt.weg2 import launcher as L

        for extra in ((), ("--dual-layout",), ("--dual-layout", "--dual-mps", "on")):
            ns = self._ns(*extra)
            assert not L.dual_priority_armed(ns)
            assert L.dual_priority_env(ns, "P") == {} and L.dual_priority_env(ns, "D") == {}
            assert L.dual_priority_front_argv(ns) == []
            L.refuse_dual_priority(ns)  # no-op

    def test_refused_outside_the_dual_layout(self):
        from sglang.srt.weg2 import launcher as L

        for extra in (("--dual-priority", "d"), ("--dual-d-capture-prio", "on"),
                      ("--dual-p-mps-low-prio", "on")):
            with pytest.raises(L.Weg2DualLayoutRefused, match="need --dual-layout"):
                L.resolve_dual_layout(self._ns(*extra))

    def test_refusals_inside_dual(self):
        from sglang.srt.weg2 import launcher as L

        for extra in (("--dual-priority", "d", "--dual-share-actuators", "mps"),
                      ("--dual-priority", "d", "--dual-p-min-share", "0"),
                      ("--dual-priority", "d", "--dual-d-min-rate-tps", "-1"),
                      ("--dual-priority", "d", "--dual-share-actuators", "duty", "--dual-p-duty", "0.5")):
            with pytest.raises(L.Weg2DualLayoutRefused):
                L.refuse_dual_priority(self._ns("--dual-layout", *extra))

    def test_each_switch_alone(self):
        from sglang.srt.weg2 import launcher as L

        ns = self._ns("--dual-layout", "--dual-d-capture-prio", "on")
        assert L.dual_priority_env(ns, "D") == {S.D_CAPTURE_PRIO_ENV: "1"}
        assert L.dual_priority_env(ns, "P") == {} and L.dual_priority_front_argv(ns) == []

        lines = []
        ns = self._ns("--dual-layout", "--dual-p-mps-low-prio", "on")
        assert L.dual_priority_env(ns, "P", lines.append) == {}
        assert lines[0].startswith("W-DUAL-SHARE-FALLBACK mech=mps_client_priority")
        ns = self._ns("--dual-layout", "--dual-mps", "on", "--dual-p-mps-low-prio", "on")
        with mock.patch.object(L, "_read_driver_text", lambda: "Kernel Module  595.58.03"):
            assert L.dual_priority_env(ns, "P") == {"CUDA_MPS_CLIENT_PRIORITY": "1"}
        assert L.dual_priority_env(ns, "D") == {}

        ns = self._ns("--dual-layout", "--dual-priority", "dynamic", "--dual-share-actuators", "chunk,duty",
                      "--dual-d-min-rate-tps", "15")
        env = L.dual_priority_env(ns, "P")
        assert env[S.CTL_ENV] == S.ctl_path("t") and env[S.ACT_ENV] == "chunk,duty" and env[S.DUTY_ENV] == "1"
        assert "CUDA_MPS_CLIENT_PRIORITY" not in env and L.dual_priority_env(ns, "D") == {}
        argv = L.dual_priority_front_argv(ns)
        assert argv[:4] == ["--dual-priority", "dynamic", "--dual-share-ctl", S.ctl_path("t")]
        assert "--dual-d-min-rate-tps" in argv and argv[argv.index("--dual-d-min-rate-tps") + 1] == "15"

    def test_ctl_sits_next_to_the_busy_file(self):
        from sglang.srt.weg2 import dual_duty as DD

        assert S.ctl_path("tag") == DD.dbusy_path("tag") + ".ctl"

    def test_main_wires_both_groups_and_the_front(self):
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L.main)
        assert ('spec_p.env.update(dual_share_env(ns, "P"))\n'
                '    spec_p.env.update(dual_priority_env(ns, "P", log))') in src
        assert src.count('dual_priority_env(ns, "D", log)') == 2
        assert "dual_priority_front_argv(ns)" in inspect.getsource(L.front_argv_for)
