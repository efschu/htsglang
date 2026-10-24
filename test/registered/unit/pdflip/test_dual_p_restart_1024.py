# SPDX-License-Identifier: Apache-2.0
"""S7b P-RESTART (24.10., Auftrag #214): in the 27B dual layout a dead P group no longer ends the boot.

Specimens (27B NVFP4 dual, front logs under /spinning/docker-acceptance/27b/evidence): 10240650 -- all three P
ranks die in the wake path (AttributeError _pdflip_wake_defer_work), D decodes on, the front stops with
``PDFLIP STOP W-DUAL-P-WAKE`` and SIGTERMs D 46 s later; 10210031 / 10220215 -- P dies at a host-side point, D
decodes on, ``W17 PdFlipGroupDead`` stops the boot and SIGTERMs D (3 min / 4.7 min later). #210 made D survive on
the card; this is the front's half (stage 1) and the P restart (stage 2).

Danger directions: (1) a P death still stops the boot in the dual layout (the specimens); (2) the flip form or a
D death is taken over (must keep the STOP); (3) D prefills over X while P is away (Phasen-Gesetz) or takes a
p_only request; (4) a P leg that died is failed to the client, or handed to D half-prefilled; (5) P counts as back
on /health alone while its ranks have not signed in (R11 barrier), or is respawned while the old session still
holds the card; (6) the restart loops without bound. Review round (review-p-restart-1024.md): (7) the card gate
passes on the argv budget while P's plan (budget + overhead) does not fit (OOM); (8) orphaned P ranks of a dead
leader survive the kill; (9) a respawned P is left standing after a failure; (10) a held rank (#1223) is killed;
(11) the front reports 200 while P is gone for good; (12) the dead P's arena refs go unnamed.

Fake front: ``object.__new__(Front)`` with the attributes the touched methods read (the house pattern of
test_pdflip_front_health_facts_b1_0927). Fake launcher: an IO double with the driver's methods.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import inspect
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import dual_p_restart as S  # noqa: E402
from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import front_health as FH  # noqa: E402

P_SID, D_SID = 4711, 4712
_FD0 = [0]


def _fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0


def setUpModule():
    _FD0[0] = _fds()


def tearDownModule():
    # the neighbour battery once died of EMFILE (24.10. 12:42Z): this file must not leak descriptors
    grown = _fds() - _FD0[0]
    assert grown <= 4, "test_dual_p_restart_1024 leaked %d file descriptors" % grown
X = 6000


def _env(survive="1", restart="0", **extra):
    """Stage-1 env: the restart off unless asked (no respawn record anyway)."""
    e = {FH.ENV: "1", S.ENV: survive, S.RESTART_ENV: restart}
    e.update(extra)
    return e


class _Env:
    def __init__(self, **kw):
        self.p = mock.patch.dict(os.environ, _env(**kw))

    def __enter__(self):
        self.p.start()
        os.environ.pop(S.SPEC_ENV, None)
        return self

    def __exit__(self, *a):
        self.p.stop()
        return False


def _front(dual=True, state="serving"):
    f = object.__new__(front_mod.Front)
    f.groups = {"P": front_mod.Group("P", "http://p", P_SID), "D": front_mod.Group("D", "http://d", D_SID)}
    f.state, f.awake, f.epoch, f.stop = state, "D", 11, None
    f.t0 = time.time() - 600
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f._ready_for_d = collections.deque()
    f._batch_gate = asyncio.Event()
    f.dual_layout = dual
    f.tp_prefill_max_tokens = X
    f._dual_inflight = {}
    return f


def _p(rid, u, *, skip=False, p_only=False):
    return types.SimpleNamespace(rid=rid, est_uncached=u, skip_leg1=skip, p_only=p_only, leg1_done=False,
                                 d_direct=False, fut=concurrent.futures.Future())


def _http(status):
    async def probe(self, g, timeout_s):
        return status[g.name]
    return probe


def _poll(front, http, alive, d):
    with mock.patch.object(front_mod.Front, "_probe_group_health", _http(http)), \
         mock.patch.object(front_mod, "_sid_alive", lambda sid: alive.get(sid, True)), \
         mock.patch.object(FH, "pid_session", lambda pid: None):
        asyncio.run(front.health_poll_once([d]))


# ------------------------------------------------------------------------------------------------ stage 1
class OnPDead(unittest.TestCase):
    def test_taken_over_only_in_the_dual_layout_with_the_switch_on(self):
        with _Env():
            self.assertTrue(S.on_p_dead(_front(), "x"))
            self.assertFalse(S.on_p_dead(_front(dual=False), "x"))          # flip form: the STOP stays
            self.assertFalse(S.on_p_dead(_front(state="STOP"), "x"))
        with _Env(survive="0"):
            self.assertFalse(S.on_p_dead(_front(), "x"))                    # the old STOP, by switch

    def test_the_line_the_state_and_the_inflight_marks(self):
        with _Env():
            f = _front()
            leg = _p("pdflip-0-6", 120000)
            f._dual_inflight = {leg.rid: leg}
            with self.assertLogs(S.logger, "ERROR") as cm:
                self.assertTrue(S.on_p_dead(f, "W17 PdFlipGroupDead group P"))
            self.assertEqual(S.state_of(f).state, S.FAILED)                  # no restart armed: gone for good
            self.assertTrue(S.state_of(f).no_restart)
            self.assertTrue(leg.dual_pause)                                   # the dual pass requeues it
            self.assertTrue(any("S7B P-LEG-ABORTED rid=pdflip-0-6" in m for m in cm.output))
            self.assertEqual(S.state_of(f).old_sid, P_SID)
            self.assertTrue(any("S7B P-DEAD why=W17 PdFlipGroupDead group P" in m for m in cm.output))
            self.assertTrue(S.p_absent(f))
            self.assertTrue(S.on_p_dead(f, "again"))                         # idempotent, still no STOP
            self.assertEqual(S.state_of(f).deaths, 1)

    def test_d_takes_what_it_may_and_never_over_x_or_p_only(self):
        with _Env():
            f = _front()
            short, halb, edge = _p("s", 300, skip=True), _p("h", 3500), _p("e", X)
            big, vis = _p("g", X + 1), _p("v", 400, p_only=True)
            f.queue.extend([short, halb, edge, big, vis])
            S.on_p_dead(f, "x")
            self.assertEqual([p.rid for p in f._ready_for_d], ["s", "h", "e"])
            self.assertTrue(short.leg1_done and not short.d_direct)          # SK SHORT: no leg 1
            self.assertTrue(halb.d_direct and edge.d_direct)
            # no restart armed: the P-bound ones are answered by name at once
            self.assertEqual(list(f.queue), [])
            for p in (big, vis):
                self.assertIsInstance(p.fut.exception(timeout=0), S.PdFlipPDead)
                self.assertIn("P-ABSENT-NO-RESTART", str(p.fut.exception(timeout=0)))
            for p in (short, halb, edge):
                self.assertFalse(p.fut.done())
            self.assertFalse(f._batch_gate.is_set())                         # the gate follows the deque

    def test_with_a_restart_armed_the_p_bound_wait(self):
        with _Env():
            f = _front()
            big = _p("g", 90000)
            f.queue.append(big)
            spawned = []
            with mock.patch.object(S, "restart_armed", lambda env=None: True):
                S.on_p_dead(f, "x", now=1000.0, spawn_driver=lambda fr: spawned.append(fr) or "task")
            self.assertEqual(S.state_of(f).state, S.RESTARTING)
            self.assertEqual(spawned, [f])
            self.assertFalse(big.fut.done())
            S.tick(f, 1000.0 + S.DEFAULT_WAIT_S - 1)
            self.assertFalse(big.fut.done())                                  # waits for P
            S.tick(f, 1000.0 + S.DEFAULT_WAIT_S + 1)
            self.assertIn("P-WAIT-EXPIRED", str(big.fut.exception(timeout=0)))


class LegAbort(unittest.TestCase):
    def _f(self):
        f = _front()
        f.dual_kv_ledgers = ["/dev/shm/wkv-test"]
        return f

    def test_a_leg_of_a_dead_p_goes_back_to_the_head_through_the_pause_path(self):
        with _Env():
            f = self._f()
            S.on_p_dead(f, "x")
            f.queue.append(_p("q", 70000))
            leg = _p("pdflip-0-6", 120000)
            leg.intake_stalled = False
            with self.assertLogs(S.logger, "ERROR") as cm:
                self.assertTrue(S.mark_failed_leg(f, leg, ConnectionResetError("peer gone")))
            self.assertTrue(leg.dual_pause)
            self.assertTrue(any("S7B P-LEG-ABORTED rid=pdflip-0-6" in m for m in cm.output))
            # what the (pinned, unchanged) dual pass does with a dual_pause leg: its own pause requeue
            front_mod.Front._dual_requeue_paused(f, leg)
            self.assertIs(f.queue[0], leg)
            self.assertFalse(leg.leg1_done)
            self.assertFalse(leg.fut.done())
            self.assertTrue(front_mod.Front._dual_skip_after_requeue(f, leg))   # on_done never hands it to D

    def test_the_leg_detects_the_death_before_the_poller(self):
        with _Env():
            f = self._f()
            leg = _p("pdflip-0-7", 120000)
            self.assertTrue(S.mark_failed_leg(f, leg, RuntimeError("500"), p_looks_dead=lambda fr: True))
            self.assertEqual(S.state_of(f).state, S.FAILED)
            self.assertTrue(leg.dual_pause)

    def test_a_live_p_the_flip_form_or_no_ledger_keep_the_old_failure(self):
        with _Env():
            f = self._f()
            leg = _p("a", 1)
            self.assertFalse(S.mark_failed_leg(f, leg, RuntimeError("x"), p_looks_dead=lambda fr: False))
            self.assertFalse(getattr(leg, "dual_pause", False))
            self.assertEqual(S.state_of(f).state, S.SERVING)
            g = _front(dual=False)
            g.dual_kv_ledgers = ["x"]
            self.assertFalse(S.mark_failed_leg(g, _p("b", 1), RuntimeError("x"), p_looks_dead=lambda fr: True))
            h = _front()                                                      # dual without ledgers: no gate
            h.dual_kv_ledgers = []
            self.assertFalse(S.mark_failed_leg(h, _p("c", 1), RuntimeError("x"), p_looks_dead=lambda fr: True))

    def test_leg1_rpc_marks_a_transport_failure_and_a_non_200(self):
        async def boom():
            raise ConnectionResetError("P gone")

        async def code(c):
            return c, b"x"

        with _Env():
            f = self._f()
            S.on_p_dead(f, "x")
            a, b, c = _p("a", 9), _p("b", 9), _p("c", 9)
            with self.assertRaises(ConnectionResetError):
                asyncio.run(S.leg1_rpc(f, a, boom()))
            self.assertTrue(a.dual_pause)
            self.assertEqual(asyncio.run(S.leg1_rpc(f, b, code(500))), (500, b"x"))
            self.assertTrue(b.dual_pause)
            self.assertEqual(asyncio.run(S.leg1_rpc(f, c, code(200))), (200, b"x"))
            self.assertFalse(getattr(c, "dual_pause", False))

    def test_the_poller_fact_is_a_dead_p(self):
        f = _front()
        f.groups["P"].health_facts = FH.GroupFacts(False, False, 1, None, time.time())
        self.assertTrue(S._p_looks_dead(f))
        f.groups["P"].health_facts = FH.GroupFacts(True, True, 0, None, time.time())
        with mock.patch.object(front_mod, "_sid_alive", lambda sid: True):
            self.assertFalse(S._p_looks_dead(f))


class FrontWiring(unittest.TestCase):
    def test_w17_on_p_in_the_dual_layout_does_not_stop(self):
        with _Env(), tempfile.TemporaryDirectory() as d:
            f = _front()
            for _ in range(2):
                _poll(f, {"P": False, "D": True}, {P_SID: False, D_SID: True}, d)
            self.assertEqual(f.state, "serving")
            self.assertIsNone(f.stop)
            self.assertEqual(S.state_of(f).state, S.FAILED)
            for _ in range(3):                                                # absent: no W17 later either
                _poll(f, {"P": False, "D": True}, {P_SID: False, D_SID: True}, d)
            self.assertEqual(f.state, "serving")

    def test_the_control_cases_keep_the_stop(self):
        with _Env(survive="0"), tempfile.TemporaryDirectory() as d:
            f = _front()
            for _ in range(2):                                                # W17's streak is 2
                _poll(f, {"P": False, "D": True}, {P_SID: False, D_SID: True}, d)
            self.assertEqual(f.state, "STOP")
            self.assertIn("W17 PdFlipGroupDead", str(f.stop))
        with _Env(), tempfile.TemporaryDirectory() as d:
            f = _front(dual=False)                                            # the flip form: unchanged
            for _ in range(2):
                _poll(f, {"P": False, "D": True}, {P_SID: False, D_SID: True}, d)
            self.assertEqual(f.state, "STOP")
        with _Env(), tempfile.TemporaryDirectory() as d:
            f = _front()                                                      # a dead D still stops
            for _ in range(2):
                _poll(f, {"P": True, "D": False}, {P_SID: True, D_SID: False}, d)
            self.assertEqual(f.state, "STOP")
            self.assertIn("group D", str(f.stop))

    def test_health_stays_200_while_d_serves(self):
        with _Env():
            f = _front()
            f.groups["D"].health_facts = FH.GroupFacts(True, True, 0, None, time.time())
            with mock.patch.object(S, "restart_armed", lambda env=None: True):
                S.on_p_dead(f, "x", spawn_driver=lambda fr: None)
            with mock.patch.object(front_mod.Front, "_probe_group_health", _http({"P": False, "D": True})):
                r = asyncio.run(f._handle_health_facts())
            body = json.loads(r.body)
            self.assertEqual(r.status, 200)                                   # P may still come back
            self.assertEqual(body["P"], "absent: S7B restarting")
            self.assertEqual(body["unhealthy"], {})

    def test_a_refused_p_wake_does_not_stop(self):
        async def leg_rpc(self, g, path, body, timeout):
            return 500, "AttributeError: _pdflip_wake_defer_work"

        for survive, want in (("1", "serving"), ("0", "STOP")):
            with _Env(survive=survive):
                f = _front()
                f.boot_epoch = 7
                with mock.patch.object(front_mod.Front, "leg_rpc", leg_rpc), \
                     mock.patch.object(front_mod.Front, "_dual_p_weights_tags", lambda self: []):
                    asyncio.run(f._dual_p_wake())
                self.assertEqual(f.state, want)
                if want == "STOP":
                    self.assertIn("W-DUAL-P-WAKE", str(f.stop))
                else:
                    self.assertIn("W-DUAL-P-WAKE", S.state_of(f).why)

    def test_controller_pass_and_leg_sites_are_wired(self):
        src = inspect.getsource(front_mod.Front.controller)
        dual = src[src.index("if self.dual_layout:\n                    if _s7b.p_absent(self)"):]
        self.assertLess(dual.index("_s7b.tick(self"), dual.index("self._dual_pump(_p_drain_pass_dual)"))
        one = src[src.index("async def _p_drain_pass_dual"):src.index("async def _p_drain_pass()")]
        self.assertNotIn("_s7b", one)                                          # the pinned 27B pass stays byte-identical
        self.assertIn("self._dual_dispatch_held()", one)                       # ... and reads the gate below per leg
        self.assertIn("_s7b.leg1_rpc(self, p, self._leg1_bounded(p, _post()))", inspect.getsource(front_mod.Front.leg1))
        self.assertEqual(len(re.findall(r"_s7b\.on_p_dead\(", inspect.getsource(front_mod.Front.health_poll_once))), 1)

    def test_the_running_pass_dispatches_no_leg_while_p_is_absent(self):
        with _Env():
            f = _front()
            f.dual_kv_ledgers = ["/dev/shm/wkv-none-s7b"]
            S.on_p_dead(f, "x")
            self.assertTrue(front_mod.Front._dual_dispatch_held(f))

    def test_state_block(self):
        with _Env():
            self.assertEqual(S.state_block(_front(dual=False)), {})
            f = _front()
            S.on_p_dead(f, "x")
            b = S.state_block(f)["p_restart"]
            self.assertEqual((b["state"], b["armed"], b["restart_armed"]), (S.FAILED, True, False))


# ------------------------------------------------------------------------------------------------ stage 2
class FakeIO:
    """The Fake-Launcher: the driver's effects as a script."""

    def __init__(self, *, alive_for=0, survives_kill=False, health_after=0, signed_after=0, exits_rc=None,
                 spec=True, stage_files=True, room_at=0, spec_env=None):
        self.t = 0.0
        self.spec = {"argv": ["python3", "-m", "x"], "env": dict(spec_env or {})} if spec else {}
        self.alive_for, self.survives_kill = alive_for, survives_kill
        self.health_after, self.signed_after, self.exits_rc = health_after, signed_after, exits_rc
        self.stage_files = stage_files
        self.room_at = room_at
        self.calls = []
        self.killed = False

    def now(self):
        return self.t

    async def sleep(self, s):
        self.t += s

    def session_alive(self, sid):
        if self.killed and not self.survives_kill:
            return False
        return self.t < self.alive_for

    async def end_session(self, sid):
        self.calls.append(("end_session", sid, self.t))
        self.killed = True
        return "pids=[1,2] mps_terminated=[1,2] kill_process_tree=ok"

    def card_room(self):
        ok = self.t >= self.room_at
        return ok, ("nvml0 free=7000>=need=6610" if ok else "nvml0 free=2100<need=6610")

    def spawn(self):
        self.calls.append(("spawn", self.t, self.session_alive(P_SID)))
        self.t_spawn = self.t
        return 9999

    def spawned_exited(self):
        return self.exits_rc

    async def health_ok(self):
        return self.t - self.t_spawn >= self.health_after

    def ranks_signed_in(self, t_spawn):
        if not self.stage_files:
            return None
        return self.t - t_spawn >= self.signed_after


def _drive(f, io, **env):
    with _Env(restart="1", **env):
        return asyncio.run(S.restart_driver(f, io))


def _dead_front():
    with _Env():
        f = _front()
        S.on_p_dead(f, "x", spawn_driver=lambda fr: None)
    st = S.state_of(f)
    st.state, st.no_restart = S.RESTARTING, False
    f._dual_stages_obj = object()
    f.groups["P"].health_fail_streak = 9
    f.groups["P"].outstanding["pdflip-0-6"] = 1.0
    return f


class RestartDriver(unittest.TestCase):
    def test_happy_path(self):
        f = _dead_front()
        io = FakeIO(alive_for=40, health_after=60, signed_after=64)
        self.assertEqual(_drive(f, io), S.SERVING)
        self.assertEqual([c[0] for c in io.calls], ["spawn"])                  # the old session ended alone
        self.assertFalse(io.calls[0][2])                                       # never spawned over a live session
        self.assertEqual(f.groups["P"].sid, 9999)
        st = S.state_of(f)
        self.assertEqual((st.state, st.restarts, st.new_pid), (S.SERVING, 1, 9999))
        self.assertIsNone(f._dual_stages_obj)                                  # a fresh P ladder
        self.assertEqual((f.groups["P"].health_fail_streak, f.groups["P"].outstanding), (0, {}))
        self.assertFalse(S.p_absent(f))

    def test_an_old_session_past_the_grace_is_ended_first(self):
        f = _dead_front()
        io = FakeIO(alive_for=10 ** 6)
        self.assertEqual(_drive(f, io), S.SERVING)
        self.assertEqual([c[0] for c in io.calls], ["end_session", "spawn"])
        self.assertGreaterEqual(io.calls[0][2], S.DEFAULT_GRACE_S)

    def test_an_unkillable_old_session_is_a_named_failure(self):
        f = _dead_front()
        big = _p("g", 90000)
        f.queue.append(big)
        io = FakeIO(alive_for=10 ** 6, survives_kill=True)
        with self.assertLogs(S.logger, "ERROR") as cm:
            self.assertEqual(_drive(f, io), S.FAILED)
        self.assertNotIn("spawn", [c[0] for c in io.calls])
        self.assertIn("P-RESTART-FAILED", str(big.fut.exception(timeout=0)))
        self.assertTrue(any("S7B P-RESTART FAILED" in m for m in cm.output))
        self.assertEqual(f.state, "serving")                                   # D serves on

    def test_health_alone_is_not_back_while_a_rank_has_not_signed_in(self):
        f = _dead_front()
        io = FakeIO(health_after=0, signed_after=10 ** 6)
        self.assertEqual(_drive(f, io, **{S.READY_ENV: "120"}), S.FAILED)
        self.assertIn("stage file", S.state_of(f).note)

    def test_no_stage_files_means_health_decides(self):
        f = _dead_front()
        self.assertEqual(_drive(f, FakeIO(health_after=4, stage_files=False)), S.SERVING)

    def test_a_respawn_that_exits_is_a_failure(self):
        f = _dead_front()
        self.assertEqual(_drive(f, FakeIO(health_after=10 ** 6, exits_rc=1)), S.FAILED)
        self.assertIn("rc=1", S.state_of(f).note)

    def test_the_budget_bounds_the_restarts(self):
        f = _dead_front()
        S.state_of(f).restarts = S.DEFAULT_MAX
        io = FakeIO()
        self.assertEqual(_drive(f, io), S.FAILED)
        self.assertEqual(io.calls, [])

    def test_the_card_room_gate_holds_the_spawn(self):
        f = _dead_front()
        io = FakeIO(room_at=200)
        self.assertEqual(_drive(f, io), S.SERVING)
        self.assertGreaterEqual(io.calls[0][1], 200)                          # spawned only once the card had room
        f = _dead_front()
        io = FakeIO(room_at=10 ** 6)
        self.assertEqual(_drive(f, io, **{S.CARD_WAIT_ENV: "60"}), S.FAILED)
        self.assertEqual(io.calls, [])
        self.assertIn("nvml0 free=2100<need=6610", S.state_of(f).note)

    def test_no_record_no_spawn(self):
        f = _dead_front()
        io = FakeIO(spec=False)
        self.assertEqual(_drive(f, io), S.FAILED)
        self.assertEqual(io.calls, [])


class CardNeed(unittest.TestCase):
    # specimen 10240650: launcher.log "D sized from P's PLAN (budget + overhead, MiB): nvml0 9110, nvml1 7550,
    # nvml2 7700" -- P's argv budget alone is 6610/5050/5200 (review MUSS 1: 2500 MiB/card short)
    PLAN = {0: 9110, 1: 7550, 2: 7700}

    def test_the_gate_checks_the_plan_not_the_argv_budget(self):
        spec = {"argv": ["--rank-gpu-id", "0,1,2", "--rank-gpu-memory-mib", "6610,5050,5200"],
                "card_need_mib": {str(k): v for k, v in self.PLAN.items()}, "env": {}}
        io = S.RealIO(_front(), spec, {})
        free = [front_mod.CardFree(nvml_index=i, uuid="GPU-%d" % i, free_mib=m, reserved_mib=0, carve_out_known=True)
                for i, m in ((0, 7000), (1, 9000), (2, 9000))]
        with mock.patch.object(front_mod, "_nvml_free", lambda: free):
            ok, why = io.card_room()
        self.assertFalse(ok)                                                  # 7000 >= 6610 but < 9110
        self.assertIn("nvml0 free=7000<need=9110", why)
        free[0] = front_mod.CardFree(nvml_index=0, uuid="GPU-0", free_mib=9110, reserved_mib=0, carve_out_known=True)
        with mock.patch.object(front_mod, "_nvml_free", lambda: free):
            self.assertTrue(io.card_room()[0])                                # exactly the plan, no reserve on top

    def test_no_plan_in_the_record_no_spawn(self):
        io = S.RealIO(_front(), {"argv": ["--rank-gpu-memory-mib", "1"], "env": {}}, {})
        ok, why = io.card_room()
        self.assertFalse(ok)
        self.assertIn("no planned P peak", why)

    def test_verdict(self):
        self.assertFalse(S.card_room_verdict(self.PLAN, {})[0])               # NVML blind: no spawn on a guess
        self.assertFalse(S.card_room_verdict({}, {0: 99999})[0])
        ok, why = S.card_room_verdict(self.PLAN, {0: 9110, 1: 7549, 2: 9000})
        self.assertFalse(ok)
        self.assertIn("nvml1 free=7549<need=7550", why)


class RespawnRecord(unittest.TestCase):
    def test_roundtrip_mode_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            path = S.respawn_path(d, "dkr27bnvfp4dualbar1drei10241200")
            S.write_spec(path, argv=["python3", "-m", "flliper.launch_server", "--port", "30001"],
                         env={"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps/pipe"}, cwd="/x", log="/x/P.log", port=30001,
                         tag="t", pid=42, card_need_mib={0: 9110})
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            rec = S.read_spec(path)
            self.assertEqual(rec["argv"][-1], "30001")
            self.assertEqual(rec["env"]["CUDA_MPS_PIPE_DIRECTORY"], "/tmp/mps/pipe")
            self.assertEqual(S.spec_card_need(rec), {0: 9110})
            with mock.patch.dict(os.environ, {S.SPEC_ENV: path, S.RESTART_ENV: "1"}):
                self.assertTrue(S.restart_armed())
            with mock.patch.dict(os.environ, {S.SPEC_ENV: path, S.RESTART_ENV: "0"}):
                self.assertFalse(S.restart_armed())
            with open(path, "w") as fh:
                json.dump({"schema": "other", "argv": ["x"], "env": {}}, fh)
            self.assertIsNone(S.read_spec(path))
            self.assertIsNone(S.read_spec(os.path.join(d, "missing.json")))

    def test_the_launcher_writes_it_for_the_dual_layout_only(self):
        from flliper.srt.pdflip import launcher as L

        with tempfile.TemporaryDirectory() as d, mock.patch.object(L, "GPU_ARB", d):
            spec = L.GroupSpec("P", 30001, ["python3", "-m", "x", "--port", "30001"], os.path.join(d, "P.log"),
                               {"A": "1"}, pid=77, cwd="/core")
            lines = []
            ns = types.SimpleNamespace(dual_layout=True, tag="t1")
            with mock.patch.dict(os.environ, {S.RESTART_ENV: "1"}):
                path = L.write_p_respawn_spec(spec, ns, lines.append, False, {0: 9110, 1: 7550})
            rec = S.read_spec(path)
            self.assertEqual((rec["argv"], rec["env"], rec["cwd"], rec["pid"]), (spec.argv, {"A": "1"}, "/core", 77))
            self.assertEqual(S.spec_card_need(rec), {0: 9110, 1: 7550})
            self.assertTrue(any("S7B P-RESPAWN-SPEC" in ln for ln in lines))
            self.assertEqual(L.write_p_respawn_spec(spec, types.SimpleNamespace(dual_layout=False, tag="t2"),
                                                    lines.append, False), "")
            with mock.patch.dict(os.environ, {S.RESTART_ENV: "0"}):           # stage 1 only: no record at all
                self.assertEqual(L.write_p_respawn_spec(spec, types.SimpleNamespace(dual_layout=True, tag="t3"),
                                                        lines.append, False, {0: 1}), "")
            self.assertFalse(os.path.exists(S.respawn_path(d, "t3")))
            self.assertEqual(L.write_p_respawn_spec(spec, ns, lines.append, True), "")
            self.assertFalse(os.path.exists(S.respawn_path(d, "t2")))


# ------------------------------------------------------------------------------------------------ review round
class EndSession(unittest.TestCase):
    """MUSS 2: every pid of P's old session, a dead leader's orphans included."""

    def test_orphans_of_a_dead_leader_are_killed_one_by_one(self):
        killed, asked = [], []
        out = S.end_session_pids(
            4711, session_pids=lambda sid: {11, 12}, terminate=lambda pids: asked.append(list(pids)) or [11, 12],
            kill_tree=lambda leader: "ok", kill_pid=killed.append, alive=lambda p: True, own_sid=1234)
        self.assertEqual(asked, [[11, 12]])                                    # MPS terminate first (#210)
        self.assertEqual(sorted(killed), [11, 12])                            # then SIGKILL each pid of the session
        self.assertIn("sigkill=[11, 12]", out)

    def test_never_the_fronts_own_session(self):
        out = S.end_session_pids(77, session_pids=lambda sid: {1, 2}, terminate=lambda p: [], kill_tree=lambda l: "",
                                 kill_pid=lambda p: self.fail("killed"), alive=lambda p: True, own_sid=77)
        self.assertIn("refused", out)

    @unittest.skipIf(shutil.which("ps") is None, "ps not in this image")
    def test_real_orphans_in_a_session_die(self):
        leader = subprocess.Popen(["bash", "-c", "sleep 300 & sleep 300 & wait"], start_new_session=True)
        sid = leader.pid
        try:
            for _ in range(100):
                if len(front_mod._session_pids(sid)) >= 3:
                    break
                time.sleep(0.05)
            os.kill(sid, signal.SIGKILL)
            leader.wait(10)                                                   # the leader is gone, its sleeps orphaned
            orphans = sorted(front_mod._session_pids(sid))
            self.assertGreaterEqual(len([p for p in orphans if S._pid_live(p)]), 2)
            io = S.RealIO(_front(), {"env": {}, "argv": ["x"]}, {})
            line = asyncio.run(io.end_session(sid))
            for _ in range(100):
                if not io.session_alive(sid):
                    break
                time.sleep(0.05)
            self.assertFalse(io.session_alive(sid), line)
        finally:
            for p in front_mod._session_pids(sid):
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass


class AfterSpawnCleanup(unittest.TestCase):
    """MUSS 3: a respawned P never stays behind a failure."""

    def _ends(self, io):
        return [c for c in io.calls if c[0] == "end_session"]

    def test_ready_bound_exit_and_stop_end_the_new_session(self):
        f = _dead_front()
        io = FakeIO(health_after=10 ** 6)
        self.assertEqual(_drive(f, io, **{S.READY_ENV: "20"}), S.FAILED)
        self.assertEqual([c[1] for c in self._ends(io)], [9999])
        self.assertIn("respawned session 9999 ended", S.state_of(f).note)
        f = _dead_front()
        io = FakeIO(health_after=10 ** 6, exits_rc=1)
        self.assertEqual(_drive(f, io), S.FAILED)
        self.assertEqual([c[1] for c in self._ends(io)], [9999])              # a half-dead tree: orphans too

        class StopIO(FakeIO):
            def spawn(self_):
                n = super().spawn()
                f.state = "STOP"                                               # D died while P was joining
                return n
        f = _dead_front()
        io = StopIO(health_after=10 ** 6)
        self.assertEqual(_drive(f, io), S.FAILED)
        self.assertEqual([c[1] for c in self._ends(io)], [9999])


class HoldBoot(unittest.TestCase):
    """MUSS 4: #1223 -- a held P rank is never killed (user order 06.09.)."""
    HOLD = {"FLLIPER_DEBUG_HOLD": "1", "FLLIPER_DEBUG_HOLD_S": "100"}

    def test_the_old_session_is_never_killed(self):
        f = _dead_front()
        io = FakeIO(alive_for=10 ** 6, spec_env=self.HOLD)
        self.assertEqual(_drive(f, io), S.FAILED)
        self.assertEqual(io.calls, [])                                        # no end_session, no spawn
        self.assertIn("not killed (#1223)", S.state_of(f).note)

    def test_the_hold_ends_by_itself_then_the_restart(self):
        f = _dead_front()
        io = FakeIO(alive_for=150, spec_env=self.HOLD)
        self.assertEqual(_drive(f, io), S.SERVING)
        self.assertEqual([c[0] for c in io.calls], ["spawn"])
        self.assertGreaterEqual(io.calls[0][1], 150)

    def test_a_failed_new_p_in_a_hold_boot_stays(self):
        f = _dead_front()
        io = FakeIO(health_after=10 ** 6, spec_env=self.HOLD)
        self.assertEqual(_drive(f, io, **{S.READY_ENV: "10"}), S.FAILED)
        self.assertNotIn("end_session", [c[0] for c in io.calls])

    def test_hold_boot_reads_both_names(self):
        self.assertEqual(S.hold_boot({"env": {"SGLANG_DEBUG_HOLD": "1"}}), (True, S.DEFAULT_HOLD_S))
        self.assertEqual(S.hold_boot({"env": {"FLLIPER_DEBUG_HOLD": "0"}})[0], False)


class FailedHealth(unittest.TestCase):
    """MUSS 5: gone for good is not healthy."""

    def _health(self, f):
        f.groups["D"].health_facts = FH.GroupFacts(True, True, 0, None, time.time())
        with mock.patch.object(front_mod.Front, "_probe_group_health", _http({"P": False, "D": True})):
            r = asyncio.run(f._handle_health_facts())
        return r.status, json.loads(r.body)

    def test_no_restart_armed_is_503_named(self):
        with _Env():
            f = _front()
            S.on_p_dead(f, "x")
            status, body = self._health(f)
        self.assertEqual(status, 503)
        self.assertIn("S7B P-ABSENT-NO-RESTART", body["unhealthy"]["P"])

    def test_a_failed_restart_is_503_named(self):
        f = _dead_front()
        _drive(f, FakeIO(spec=False))
        with _Env(restart="1"):
            status, body = self._health(f)
        self.assertEqual(status, 503)
        self.assertIn("S7B P-RESTART FAILED: no respawn record", body["unhealthy"]["P"])


class ReapRanksSecrets(unittest.TestCase):
    """SOLL 3/4/5/6/7."""

    def test_the_wait_is_per_request(self):
        with _Env():
            f = _front()
            with mock.patch.object(S, "restart_armed", lambda env=None: True):
                S.on_p_dead(f, "x", now=1000.0, spawn_driver=lambda fr: None)
            old, late = _p("old", 90000), _p("late", 90000)
            old.t_arrive, late.t_arrive = 900.0, 1000.0 + S.DEFAULT_WAIT_S - 5
            f.queue.extend([old, late])
            S.tick(f, 1000.0 + S.DEFAULT_WAIT_S + 1)
            self.assertIn("P-WAIT-EXPIRED", str(old.fut.exception(timeout=0)))
            self.assertFalse(late.fut.done())                                 # arrived late: its own clock
        self.assertGreaterEqual(S.DEFAULT_WAIT_S, S.DEFAULT_GRACE_S + 30 + S.DEFAULT_CARD_WAIT_S + S.DEFAULT_READY_S)

    def test_a_respawned_leader_is_reaped_before_the_verdict(self):
        polled = []
        f = _front()
        f._s7b_proc = types.SimpleNamespace(poll=lambda: polled.append(1) or 0)
        f.groups["P"].health_facts = FH.GroupFacts(True, True, 0, None, time.time())
        with mock.patch.object(front_mod, "_sid_alive", lambda sid: False):
            self.assertTrue(S._p_looks_dead(f))
        self.assertEqual(polled, [1])
        polled.clear()
        with mock.patch.object(front_mod, "_session_pids", lambda sid: set()):
            self.assertFalse(S.RealIO(f, {"env": {}}, {}).session_alive(99))
        self.assertEqual(polled, [1])

    def test_ranks_from_pp_size(self):
        self.assertEqual(S.p_ranks(["--pp-size", "4"], 3), 4)
        self.assertEqual(S.p_ranks(["--pp-size=3", "--tp-size", "2"], 3), 6)
        self.assertEqual(S.p_ranks(["--x"], 3), 3)
        from flliper.srt.pdflip import dual_p_kv_stage as PK

        with tempfile.TemporaryDirectory() as d:
            f = _front()
            f.dual_kv_ledgers = ["x"]
            f.tag = "t"
            io = S.RealIO(f, {"env": {}, "argv": ["--pp-size", "4"], "tag": "t"}, {})
            t0 = time.time() - 10
            with mock.patch.object(PK, "stage_file", lambda tag, r, root="/dev/shm": os.path.join(d, "%s-%d" % (tag, r))):
                for r in range(3):
                    open(os.path.join(d, "t-%d" % r), "w").close()
                self.assertFalse(io.ranks_signed_in(t0))                       # PP=4: three files are not all
                open(os.path.join(d, "t-3"), "w").close()
                self.assertTrue(io.ranks_signed_in(t0))

    def test_secrets_masked_and_put_back(self):
        argv = ["python", "-m", "x", "--admin-api-key=s3cr3t", "--port", "30031", "--admin-api-key", "s3cr3t"]
        env = {"HF_TOKEN": "hf_abc", "FLLIPER_HICACHE_BIGRAM_KEYS": "1", "A": "b"}
        m_argv, m_env = S.mask_argv(argv), S.mask_env(env)
        self.assertNotIn("s3cr3t", json.dumps([m_argv, m_env]))
        self.assertNotIn("hf_abc", json.dumps([m_argv, m_env]))
        self.assertEqual(m_env["FLLIPER_HICACHE_BIGRAM_KEYS"], "1")            # a switch named *KEYS is no secret
        argv2, env2, missing = S.unmask({"argv": m_argv, "env": m_env}, {"HF_TOKEN": "hf_abc"}, "s3cr3t")
        self.assertEqual(argv2, argv)
        self.assertEqual(env2["HF_TOKEN"], "hf_abc")
        self.assertEqual(missing, [])
        _, env3, missing = S.unmask({"argv": ["x"], "env": m_env}, {}, "")
        self.assertNotIn("HF_TOKEN", env3)
        self.assertEqual(missing, ["HF_TOKEN"])

    def test_the_record_dies_with_the_boot(self):
        with tempfile.TemporaryDirectory() as d:
            path = S.respawn_path(d, "t")
            open(path, "w").close()
            self.assertTrue(S.remove_spec(d, "t").startswith("removed"))
            self.assertEqual(S.remove_spec(d, "t"), "none")
        from flliper.srt.pdflip import launcher as L

        self.assertIn("_s7b.remove_spec(GPU_ARB", inspect.getsource(L.teardown))

    def test_the_record_never_holds_a_secret_in_clear(self):
        with tempfile.TemporaryDirectory() as d:
            path = S.write_spec(os.path.join(d, "r.json"), argv=["x", "--admin-api-key=s3cr3t"],
                                env={"HF_TOKEN": "hf_abc", "A": "1"}, cwd="/", log="/l", port=1, tag="t", pid=2)
            raw = open(path).read()
        self.assertNotIn("s3cr3t", raw)
        self.assertNotIn("hf_abc", raw)

    def test_spawn_puts_secrets_back_and_names_the_new_p_for_teardown(self):
        from flliper.srt.pdflip import launcher as L

        with tempfile.TemporaryDirectory() as d, mock.patch.object(L, "GPU_ARB", d), \
                mock.patch.object(L, "EVIDENCE_DIR", d), mock.patch.dict(os.environ, {"S7B_TEST_TOKEN": "tok"}):
            boot = L._operator_dir(d, "boot_t.json")
            os.makedirs(os.path.dirname(boot), exist_ok=True)
            with open(boot, "w") as fh:
                json.dump({"pids": {"P": 11, "D": 12}, "tag": "t"}, fh)
            out = os.path.join(d, "env.txt")
            code = "import os,sys; open(sys.argv[1],'w').write(os.environ.get('S7B_TEST_TOKEN','-')); import time; time.sleep(30)"
            spec = {"argv": [sys.executable, "-c", code, out], "env": S.mask_env({"S7B_TEST_TOKEN": "tok", "PATH": os.environ.get("PATH", "")}),
                    "log": os.path.join(d, "P.log"), "tag": "t", "cwd": d}
            f = _front()
            io = S.RealIO(f, spec, {})
            pid = io.spawn()
            try:
                self.assertEqual(json.load(open(boot))["pids"]["P"], pid)
                self.assertIsNone(io.spawned_exited())
                for _ in range(100):
                    if os.path.exists(out) and open(out).read():
                        break
                    time.sleep(0.05)
                self.assertEqual(open(out).read(), "tok")                     # the masked value came back
                self.assertIn("RE-launched by the front (S7B)", open(spec["log"]).read())
            finally:
                os.kill(pid, signal.SIGKILL)
                f._s7b_proc.wait(10)

    def test_boot_json_names_the_new_p(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "boot_t.json")
            with open(path, "w") as fh:
                json.dump({"pids": {"P": 11, "D": 12, "front": 13}, "tag": "t"}, fh)
            self.assertIn("11 -> 99", S.rewrite_boot_pids(path, "P", 99))
            self.assertEqual(json.load(open(path))["pids"], {"P": 99, "D": 12, "front": 13})


class ArenaCensus(unittest.TestCase):
    """MUSS 6: the dead P's arena state is measured at P-DEAD and named at P-BACK."""

    def test_p_back_names_the_difference(self):
        snaps = iter([{"arena-32768.bin": {"slots": 112, "complete": 90, "pinned": 50, "refs": 179, "claimed": 3}},
                      {"arena-32768.bin": {"slots": 112, "complete": 92, "pinned": 48, "refs": 120, "claimed": 1}}])
        with mock.patch.object(S, "ARENA_CENSUS", lambda fr: next(snaps)):
            f = _dead_front()
            with self.assertLogs(S.logger, "WARNING") as cm:
                self.assertEqual(_drive(f, FakeIO()), S.SERVING)
        line = [m for m in cm.output if "S7B P-BACK" in m][0]
        self.assertIn("arena-32768.bin:pinned 50->48 refs 179->120 claimed 3->1", line)

    def test_delta_text(self):
        self.assertEqual(S.arena_delta_text({}, {}), "none")
        self.assertIn("unreadable", S.arena_delta_text({"a": None}, {"a": None}))


class RealDualPass(unittest.TestCase):
    """SOLL 8: the REAL _p_drain_pass_dual (its source, byte for byte, out of Front.controller) with a dead P."""

    def _pass(self, front):
        src = inspect.getsource(front_mod.Front.controller)
        a = src.index("        async def _p_drain_pass_dual() -> int:")
        b = src.index("        async def _p_drain_pass() -> int:")
        ns = dict(front_mod.__dict__)
        ns.update(self=front, sem=asyncio.Semaphore(4), _ahead=0, _ro_state=None, _ro_max=0)  # one leg at a time
        exec(compile(textwrap.dedent(src[a:b]), "dual_pass", "exec"), ns)
        return ns["_p_drain_pass_dual"]

    def _front(self):
        f = _front()
        f.tp_prefill_max_tokens = 1                                           # the dual's X (specimen 10240650)
        f.dual_kv_ledgers = ["/dev/shm/wkv-none-s7b"]
        f.p_phase_max_requests, f.p_pool_tokens, f.p_concurrency = 0, 0, 1
        f._p_phase_k, f._p_cost_tasks, f._p_intake_stalled = [], set(), False
        return f

    def _patches(self, leg1):
        F = front_mod.Front
        return [mock.patch.object(F, "leg1", leg1), mock.patch.object(F, "_log_p_prefix_reuse", lambda self, r: None),
                mock.patch.object(F, "note_x_sample", lambda self, kind, value: None),
                mock.patch.object(front_mod, "_dual_p_pressure", lambda paths: 0),
                mock.patch.object(F, "_dual_p_held_by_stage", lambda self: False),
                mock.patch.object(F, "_q696_stall_held", lambda self: False),
                mock.patch.object(F, "_dual_resume_held", lambda self: False),
                mock.patch.object(F, "_dual_short_reorder", lambda self, head_blocked: False),
                mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_ENABLE_X_COST_LINE": "0"})]

    def _run(self, f, leg1):
        async def go():
            loop = asyncio.get_running_loop()
            mk = lambda rid, u: front_mod.Pending(rid, "/generate", {}, "t", time.time(), loop.create_future(),
                                                  est_prompt=u, est_uncached=u)
            f.queue.extend([mk("pdflip-0-6", 120000), mk("pdflip-0-7", 50000)])
            await self._pass(f)()
            return list(f.queue)
        ps = self._patches(leg1)
        for p_ in ps:
            p_.start()
        try:
            return asyncio.run(go())
        finally:
            for p_ in ps:
                p_.stop()

    def test_a_leg_that_dies_with_p_goes_back_to_the_head(self):
        async def leg1(self, p):
            async def post():
                raise ConnectionResetError("Cannot connect to host 127.0.0.1:30031")
            await S.leg1_rpc(self, p, post())

        with _Env():
            f = self._front()
            f.groups["P"].health_facts = FH.GroupFacts(False, False, 2, None, time.time())   # the poller's dead fact
            q = self._run(f, leg1)
            self.assertEqual([p.rid for p in q], ["pdflip-0-6"])               # requeued at the head, not failed
            self.assertFalse(q[0].fut.done())
            self.assertEqual(list(f._ready_for_d), [])                         # never handed to D
            self.assertEqual(f.counters["leg1_failures"], 0)
            self.assertEqual(f.counters["s7b_legs_aborted"], 1)
            self.assertEqual(S.state_of(f).state, S.FAILED)                    # no restart armed in this test
            S.tick(f)                                                          # the controller: gone for good -> named 503
            self.assertTrue(q[0].fut.done())                                   # an asyncio future: done, then read
            self.assertIn("P-ABSENT-NO-RESTART", str(q[0].fut.exception()))

    def test_a_marked_leg_that_still_returns_200_goes_to_d(self):
        async def leg1(self, p):
            p.dual_pause = True                                                # marked at the death, P had finished it
            p.leg1_prompt_tokens = 120000

        with _Env():
            f = self._front()
            q = self._run(f, leg1)
            self.assertEqual([p.rid for p in f._ready_for_d][:1], ["pdflip-0-6"])
            self.assertFalse(getattr(f._ready_for_d[0], "dual_pause", False))
            self.assertFalse(f._ready_for_d[0].fut.done())

if __name__ == "__main__":
    unittest.main()
