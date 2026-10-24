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
holds the card; (6) the restart loops without bound.

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
import stat
import tempfile
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
            self.assertEqual(S.state_of(f).state, S.DEAD)
            self.assertTrue(leg.s7b_p_dead)
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
                self.assertIsInstance(p.fut.exception(), S.PdFlipPDead)
                self.assertIn("P-ABSENT-NO-RESTART", str(p.fut.exception()))
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
            self.assertIn("P-WAIT-EXPIRED", str(big.fut.exception()))


class RequeueFailedLeg(unittest.TestCase):
    def test_a_leg_of_a_dead_p_goes_back_to_the_head(self):
        with _Env():
            f = _front()
            S.on_p_dead(f, "x")
            other = _p("q", 70000)
            f.queue.append(other)
            leg = _p("pdflip-0-6", 120000)
            with self.assertLogs(S.logger, "ERROR") as cm:
                self.assertTrue(S.requeue_failed_leg(f, leg, ConnectionResetError("peer gone")))
            self.assertIs(f.queue[0], leg)
            self.assertTrue(leg.dual_requeued)                                 # on_done never hands it to D
            self.assertFalse(leg.leg1_done)
            self.assertFalse(leg.fut.done())
            self.assertTrue(any("S7B P-LEG-ABORTED rid=pdflip-0-6" in m for m in cm.output))

    def test_the_leg_detects_the_death_before_the_poller(self):
        with _Env():
            f = _front()
            leg = _p("pdflip-0-7", 120000)
            self.assertTrue(S.requeue_failed_leg(f, leg, RuntimeError("500"), p_looks_dead=lambda fr: True))
            self.assertEqual(S.state_of(f).state, S.DEAD)
            self.assertIs(f.queue[0], leg)

    def test_a_live_p_or_the_flip_form_keeps_the_old_failure(self):
        with _Env():
            f = _front()
            self.assertFalse(S.requeue_failed_leg(f, _p("a", 1), RuntimeError("x"), p_looks_dead=lambda fr: False))
            self.assertEqual(S.state_of(f).state, S.SERVING)
            g = _front(dual=False)
            self.assertFalse(S.requeue_failed_leg(g, _p("b", 1), RuntimeError("x"), p_looks_dead=lambda fr: True))

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
            self.assertEqual(S.state_of(f).state, S.DEAD)
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
            S.on_p_dead(f, "x")
            with mock.patch.object(front_mod.Front, "_probe_group_health", _http({"P": False, "D": True})):
                r = asyncio.run(f._handle_health_facts())
            body = json.loads(r.body)
            self.assertEqual(r.status, 200)
            self.assertEqual(body["P"], "absent: S7B dead")
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
        self.assertIn("and not _s7b.p_absent(self)),", src)                    # the running pass stops dispatching
        one = src[src.index("async def _p_drain_pass_dual"):src.index("async def _p_drain_pass()")]
        self.assertLess(one.index("_s7b.requeue_failed_leg(self, p, e)"), one.index('self.counters["leg1_failures"]'))
        self.assertEqual(len(re.findall(r"_s7b\.on_p_dead\(", inspect.getsource(front_mod.Front.health_poll_once))), 1)

    def test_state_block(self):
        with _Env():
            self.assertEqual(S.state_block(_front(dual=False)), {})
            f = _front()
            S.on_p_dead(f, "x")
            b = S.state_block(f)["p_restart"]
            self.assertEqual((b["state"], b["armed"], b["restart_armed"]), (S.DEAD, True, False))


# ------------------------------------------------------------------------------------------------ stage 2
class FakeIO:
    """The Fake-Launcher: the driver's effects as a script."""

    def __init__(self, *, alive_for=0, survives_kill=False, health_after=0, signed_after=0, exits_rc=None,
                 spec=True, stage_files=True, room_at=0):
        self.t = 0.0
        self.spec = {"argv": ["python3", "-m", "x"], "env": {}} if spec else {}
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
    st.state = S.RESTARTING
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
        self.assertIn("P-RESTART-FAILED", str(big.fut.exception()))
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
    # specimen 10240650, P's argv: two --rank-gpu-memory-mib, argparse keeps the last
    ARGV = ["python", "-m", "flliper.launch_server", "--rank-gpu-id", "0,1,2", "--rank-gpu-memory-mib",
            "26256,16128,15384", "--port", "30031", "--rank-gpu-memory-mib", "6610,5050,5200"]

    def test_the_last_budget_per_card(self):
        self.assertEqual(S.card_need_from_argv(self.ARGV), {0: 6610, 1: 5050, 2: 5200})
        self.assertEqual(S.card_need_from_argv(["--rank-gpu-id=0,0,1", "--rank-gpu-memory-mib=100,200,300"]),
                         {0: 300, 1: 300})                                    # co-located ranks add up
        self.assertEqual(S.card_need_from_argv(["--rank-gpu-id", "0,1"]), {})
        self.assertEqual(S.card_need_from_argv(["--rank-gpu-id", "0,1", "--rank-gpu-memory-mib", "1,2,3"]), {})

    def test_verdict_no_reserve_on_top(self):
        need = {0: 6610, 1: 5050}
        self.assertTrue(S.card_room_verdict(need, {0: 6610, 1: 5050})[0])     # exactly the budget is enough
        ok, why = S.card_room_verdict(need, {0: 6609, 1: 9000})
        self.assertFalse(ok)
        self.assertIn("nvml0 free=6609<need=6610", why)
        self.assertFalse(S.card_room_verdict(need, {})[0])                   # NVML blind: no spawn on a guess
        self.assertTrue(S.card_room_verdict({}, {})[0])


class RespawnRecord(unittest.TestCase):
    def test_roundtrip_mode_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            path = S.respawn_path(d, "dkr27bnvfp4dualbar1drei10241200")
            S.write_spec(path, argv=["python3", "-m", "flliper.launch_server", "--port", "30001"],
                         env={"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps/pipe"}, cwd="/x", log="/x/P.log", port=30001,
                         tag="t", pid=42)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            rec = S.read_spec(path)
            self.assertEqual(rec["argv"][-1], "30001")
            self.assertEqual(rec["env"]["CUDA_MPS_PIPE_DIRECTORY"], "/tmp/mps/pipe")
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
            path = L.write_p_respawn_spec(spec, ns, lines.append, False)
            rec = S.read_spec(path)
            self.assertEqual((rec["argv"], rec["env"], rec["cwd"], rec["pid"]), (spec.argv, {"A": "1"}, "/core", 77))
            self.assertTrue(any("S7B P-RESPAWN-SPEC" in ln for ln in lines))
            self.assertEqual(L.write_p_respawn_spec(spec, types.SimpleNamespace(dual_layout=False, tag="t2"),
                                                    lines.append, False), "")
            self.assertEqual(L.write_p_respawn_spec(spec, ns, lines.append, True), "")
            self.assertFalse(os.path.exists(S.respawn_path(d, "t2")))


if __name__ == "__main__":
    unittest.main()
