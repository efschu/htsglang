"""27B Dual round 3 (boot dkr27bnvfp4dualbar1drei10220215, 884848bc86): stage-2 sleep, then a wake before the sleep leg.

front.log 02:35:55,074 ``DUAL-KV-PRESSURE stage=2 ... p_state=sleeping`` (the ladder enters 'sleeping' when it ISSUES the
sleep) -> the sleep leg's quiesce polls ``/flush_cache`` (400, 400, ... P not idle; no ``leg=sleep`` RPC ever issued) ->
02:35:59,018 ``D-REFILL ... freed_by=leg2_finished`` (a D seat ends) -> 02:35:59,143 ``stage=resume ... from=sleep`` ->
``/resume_memory_occupation`` on a P that parked nothing -> P.log 02:36:00 ``W29 PdFlipRankDisagree
tag=resume_memory_occupation ... KeyError: 'weights_0'`` on PP0/PP1/PP2 -> RANK-DEATH.

The wake's ``sleep_leg_done`` guard existed only under #1956 WAKE-SEES-LOAN (default off). Fix: every wake from sleep
waits for the sleep leg's answer; a quiesce refusal returns to 'stopped' (no leg pending).
"""

import asyncio
import collections
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.pdflip import dual_d_priority as DP
from flliper.srt.pdflip import front as FR
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GiB = 1 << 30
STEP_B = 64 << 20


class _PGroup:
    """Three P ranks with a torch-memory-saver-like tag table: release parks the tags on EVERY rank, resume of a
    tag a rank never parked raises KeyError there (the weight_updater.resume_memory_occupation failure) -- the
    group answers 200 only when every rank agrees (the PDFLIP-GROUP-FENCE)."""

    def __init__(self):
        self.paused = [set(), set(), set()]
        self.calls = []

    def rpc(self, path, payload):
        self.calls.append(path)
        tags = list(payload.get("tags") or [])
        errs = []
        for r in range(3):
            try:
                if path == "/release_memory_occupation":
                    self.paused[r].update(tags)
                elif path == "/resume_memory_occupation":
                    for t in tags:
                        self.paused[r].remove(t) if t in self.paused[r] else (_ for _ in ()).throw(KeyError(t))
            except KeyError as exc:
                errs.append("rank %d: KeyError: %s" % (r, exc))
        assert self.paused[0] == self.paused[1] == self.paused[2], "rank agreement"
        return (200, "ok") if not errs else (500, "W29 PdFlipRankDisagree " + ", ".join(errs))


def _front(monkeypatch, group, quiesce_gate):
    reading = {"d_short": 0}
    stops = []
    f = types.SimpleNamespace(groups={"P": "P"}, counters=collections.Counter(), weight_chunks=8, boot_epoch=1,
                              queue=collections.deque(), _d_seats_live={"seat"}, _ready_for_d=collections.deque())
    for name in ("_dual_stage_tick", "_dual_stages", "_dual_p_sleep", "_dual_p_wake", "_dual_p_weights_tags"):
        setattr(f, name, types.MethodType(getattr(FR.Front, name), f))
    f._dual_p_sleep_probe_tick = lambda pressure: False
    f._dual_p_stage_reading = lambda: {"p_committed": 0, "free_min": 20 * GiB, "p_grant_bytes": STEP_B,
                                       "d_air_bytes": STEP_B, "weights_bytes": 1 * GiB, "p_lent": 0,
                                       "card_room": [(20 * GiB, 2 * GiB)] * 3, "d_short": reading["d_short"]}

    async def quiesce(_g):
        await quiesce_gate.wait()
        return True, ""

    async def leg_rpc(_g, path, payload, _timeout):
        return group.rpc(path, payload)

    async def _lend(*_a, **_k):
        return None

    f.quiesce, f.leg_rpc, f._dual_p_lend = quiesce, leg_rpc, _lend
    f.do_stop = lambda code, why: stops.append((code, why))
    monkeypatch.setattr(FR, "_mem_available_bytes", lambda: 100 * GiB)
    monkeypatch.setenv(DP.P_SLEEP_ENV, "1")
    for k in ("FLLIPER_PDFLIP_DUAL_WAKE_SEES_LOAN", DP.SLEEP_AFTER_TICKS_ENV):
        monkeypatch.delenv(k, raising=False)
    return f, reading, stops


async def _ticks(f, n):
    for _ in range(n):
        f._dual_stage_tick(0)
        await asyncio.sleep(0)


def test_round3_seat_end_during_the_sleep_leg_does_not_wake_p(monkeypatch):
    async def go():
        group, gate = _PGroup(), asyncio.Event()
        f, reading, stops = _front(monkeypatch, group, gate)
        reading["d_short"] = 341835776                      # 02:35:54 PRESSURE-READ per_card=[341835776, 0, 0]
        await _ticks(f, 8)                                  # stop -> lend -> (held) -> sleep issued
        st = f._dual_stages_obj
        assert st.p_state == "sleeping" and f.counters["dual_kv_pressure_stage2"] == 1
        assert group.calls == [], "the quiesce still polls: nothing released yet"
        reading["d_short"] = 0
        f.counters["d_seat_done"] += 1                      # 02:35:59,018 D-REFILL freed_by=leg2_finished
        await _ticks(f, 5)
        assert "/resume_memory_occupation" not in group.calls, "no wake before the sleep leg answered"
        assert st.p_state == "sleeping" and stops == []
        gate.set()                                          # the quiesce succeeds, the release goes out
        await _ticks(f, 6)                                  # leg answered, no pressure, a seat ended -> wake
        assert group.calls == ["/release_memory_occupation", "/resume_memory_occupation"], group.calls
        assert all(not p for p in group.paused), "every rank resumed exactly what it parked"
        assert stops == [] and st.p_state == "serving"
    asyncio.run(go())


def test_quiesce_refusal_returns_to_stopped_and_releases_nothing(monkeypatch):
    async def go():
        group, gate = _PGroup(), asyncio.Event()
        f, reading, stops = _front(monkeypatch, group, gate)

        async def refuse(_g):
            return False, "flush_cache 400 for 90 s"

        f.quiesce = refuse
        reading["d_short"] = 341835776
        await _ticks(f, 5)                                  # stop, lend, held x2, sleep issued
        await asyncio.sleep(0)                              # the leg runs: quiesce refused
        st = f._dual_stages_obj
        assert st.p_state == "stopped" and st.sleep_leg_done and f.counters["dual_kv_pressure_sleep_not_idle"] >= 1
        assert "/release_memory_occupation" not in group.calls and "/resume_memory_occupation" not in group.calls
        assert stops == []
    asyncio.run(go())


def test_pure_rule_no_wake_while_the_leg_is_pending():
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    kw = dict(p_committed=0, free_min=20 * GiB, p_grant_bytes=STEP_B, d_air_bytes=STEP_B, weights_bytes=GiB,
              card_room=[(20 * GiB, 2 * GiB)] * 3)
    assert st.tick(pressure=1, seats_done=0, **kw)[0] == "stop"
    assert st.tick(pressure=1, seats_done=0, **kw)[0] == "lend"
    assert st.tick(pressure=1, seats_done=0, **kw)[0] == "sleep"
    assert st.p_state == "sleeping" and st.sleep_leg_done is False
    assert st.tick(pressure=0, seats_done=1, **kw) == (None, None)
    assert st.tick(pressure=0, seats_done=1, d_idle=True, **kw) == (None, None)
    st.sleep_leg_done = True                                # the front: release RPC 200
    assert st.tick(pressure=0, seats_done=1, **kw)[0] == "wake"


# -- side finding: the boot-order ctl read (P.log 02:18:51 PP0 vs front.log 02:18:56 controller armed) ----------------

def test_ctl_not_written_yet_is_the_start_state_not_a_fallback(tmp_path):
    from flliper.srt.pdflip import dual_share as DS

    path = str(tmp_path / "wdb.ctl")
    lines = []
    clock = [0.0]
    r = DS.CtlReader(path, log=lines.append, clock=lambda: clock[0], card="31d7ef41")
    st = r.read()
    assert st.rung == 0 and not st.ok and lines == [], "before the front's first rung: no W-DUAL-SHARE-FALLBACK"
    with open(path, "w") as f:
        f.write("v1 rung=2 f=0.5 b=1 mode=dynamic seq=1\n")
    clock[0] += 10.0
    st = r.read()
    assert st.ok and st.rung == 2
    os.unlink(path)                                          # the front's file vanishes AFTER it wrote: a real fallback
    clock[0] += 10.0
    st = r.read()
    assert st.rung == 0 and len(lines) == 1 and DS.FALLBACK in lines[0] and "FileNotFoundError" in lines[0]
