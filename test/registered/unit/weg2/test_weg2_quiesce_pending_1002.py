# SPDX-License-Identifier: Apache-2.0
"""QUIESCE-PENDING (02.10.): the P>D quiesce re-polls at once while PP0's idle
vote is on the ring, instead of sleeping the poll interval.

N5t epoch 11 (..._012d1a161a_1002_153611): the front polled /flush_cache on P
four times (400 at 51.134 / 51.236 / 51.325, 200 at 51.424). PP2's last
PASS-TAIL came at 51.319, the idle lap was home at ~51.327, but the fourth poll
went out only at 51.335: the 10 ms interval after an answer that was
"GROUP VERDICT PENDING" -- undecided, and decided by the NEXT poll. The HTTP
body could not say so: the immediate flush answered every refusal with an
empty message ("Flush cache failed.").

* rank side: a refused immediate flush whose #1268 verdict is PENDING says so
  in the 400 body (``SGLANG_WEG2_FLUSH_PENDING_MESSAGE``, default on);
* front side: after such a body the next poll goes out after
  ``SGLANG_WEG2_QUIESCE_PENDING_POLL_MS`` (default 1 ms; never longer than the
  interval), only where PP0 never re-wants a lap still on the ring (H111 /
  SGLANG_WEG2_IDLE_VOTE_NO_REWANT). Marker ``WEG2-QUIESCE-PENDING``.

#1268 MUST NOT BREAK -- pinned on the H77 harness (every real
``Scheduler._weg2_vote_*`` hook, the real ``group_idle_verdict`` /
``flush_cache`` / flush wrapper) driven by the REAL ``Front.quiesce``: polls
during the lap stamp no second lap, no lap comes home stale, exactly one
group flush, the 200 on the first poll after the lap is home; a busy rank's
NOT IDLE keeps the poll interval (no hammering) and never answers 200;
undecided never ends the quiesce. Hermetic, CPU.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import flush_wrapper as FW
from sglang.srt.weg2 import front as front_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _harness():
    sibling = Path(__file__).with_name("test_weg2_idle_round_fresh_1268.py")
    spec = importlib.util.spec_from_file_location("_h77_harness_pending", sibling)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _harness()
PENDING_BODY = FW.PENDING_MESSAGE
BUSY_BODY = "Flush cache failed.\n"


def _run(answers, fast=True, **over):
    """The real Front.quiesce against scripted (code, body) answers."""
    seq = list(answers)
    sleeps = []

    async def rpc(_g, path, _body, _timeout):
        assert path == "/flush_cache"
        return seq.pop(0)

    async def fake_sleep(s):
        sleeps.append(round(s, 4))

    fake = SimpleNamespace(rpc=rpc, _health_inflight={})
    ctx = [envs.SGLANG_WEG2_QUIESCE_FAST.override(fast)]
    for k, v in over.items():
        ctx.append(getattr(envs, k).override(v))
    with mock.patch.object(front_mod.asyncio, "sleep", new=fake_sleep):
        for c in ctx:
            c.__enter__()
        try:
            ok = asyncio.run(front_mod.Front.quiesce(fake, SimpleNamespace(name="P")))
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
    return ok, sleeps


class FrontCadence(CustomTestCase):
    def test_the_marks_are_the_schedulers_words(self):
        self.assertEqual(front_mod.QUIESCE_PENDING_MARK, FW.PENDING_MARK)
        self.assertIn(FW.PENDING_MARK, FW.PENDING_MESSAGE)

    def test_red_a_pending_answer_is_re_polled_at_once(self):
        with self.assertLogs(front_mod.logger, level="INFO") as logs:
            ok, sleeps = _run([(400, PENDING_BODY), (400, PENDING_BODY), (200, "ok")])
        self.assertEqual(ok, (True, "ok"))
        self.assertEqual(sleeps, [0.001, 0.001])
        self.assertTrue(any("WEG2-QUIESCE-PENDING group=P pending_polls=2 interval_ms=1" in m
                            for m in logs.output), logs.output)

    def test_a_busy_answer_keeps_the_interval(self):
        ok, sleeps = _run([(400, BUSY_BODY), (400, PENDING_BODY), (400, BUSY_BODY), (200, "ok")])
        self.assertEqual(sleeps, [0.01, 0.001, 0.01])

    def test_switch_zero_is_the_interval(self):
        ok, sleeps = _run([(400, PENDING_BODY), (200, "ok")],
                          SGLANG_WEG2_QUIESCE_PENDING_POLL_MS=0)
        self.assertEqual(sleeps, [0.01])

    def test_never_longer_than_the_interval(self):
        ok, sleeps = _run([(400, PENDING_BODY), (200, "ok")],
                          SGLANG_WEG2_QUIESCE_PENDING_POLL_MS=50)
        self.assertEqual(sleeps, [0.01])

    def test_armed_only_with_the_no_rewant_guard(self):
        # slow form (50 ms) with the guard: re-poll at once; without it: the interval
        ok, sleeps = _run([(400, PENDING_BODY), (200, "ok")], fast=False)
        self.assertEqual(sleeps, [0.001])
        ok, sleeps = _run([(400, PENDING_BODY), (200, "ok")], fast=False,
                          SGLANG_WEG2_IDLE_VOTE_NO_REWANT=False)
        self.assertEqual(sleeps, [0.05])

    def test_undecided_never_ends_the_quiesce(self):
        polls = []

        async def rpc(_g, path, _body, _timeout):
            polls.append(1)
            return 400, PENDING_BODY

        async def fake_sleep(s):
            await asyncio.sleep(0) if False else None

        real_sleep = asyncio.sleep

        async def tick(s):
            await real_sleep(0.002)

        fake = SimpleNamespace(rpc=rpc, _health_inflight={})
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(True), \
                mock.patch.object(front_mod, "QUIESCE_DEADLINE_S", 0.05), \
                mock.patch.object(front_mod.asyncio, "sleep", new=tick):
            ok = asyncio.run(front_mod.Front.quiesce(fake, SimpleNamespace(name="P")))
        self.assertEqual(ok, (False, PENDING_BODY), "a PENDING answer must never pass the quiesce")
        self.assertGreater(len(polls), 1)


class WrapperMessage(CustomTestCase):
    def _wrapper(self, success, detail, raises=False):
        def det():
            if raises:
                raise RuntimeError("x")
            return detail

        return FW.SchedulerFlushWrapper(flush_cache=lambda **kw: success, is_fully_idle=lambda: True,
                                        ipc_channels=None, refusal_detail=det)

    def test_a_pending_refusal_names_itself_others_stay_empty(self):
        from sglang.srt.managers.io_struct import FlushCacheReqInput

        out = self._wrapper(False, "GROUP VERDICT PENDING: the idle vote is on the ring").handle(
            FlushCacheReqInput())
        self.assertEqual((out.success, out.message), (False, FW.PENDING_MESSAGE))
        out = self._wrapper(False, "GROUP NOT IDLE: rank 2 blockers=[x]").handle(FlushCacheReqInput())
        self.assertEqual(out.message, "")
        out = self._wrapper(True, "GROUP VERDICT PENDING").handle(FlushCacheReqInput())
        self.assertEqual((out.success, out.message), (True, ""))
        out = self._wrapper(False, "", raises=True).handle(FlushCacheReqInput())
        self.assertEqual(out.message, "")
        with mock.patch.dict("os.environ", {FW.ENV_PENDING_MESSAGE: "0"}):
            out = self._wrapper(False, "GROUP VERDICT PENDING").handle(FlushCacheReqInput())
        self.assertEqual(out.message, "")

    def test_the_scheduler_wires_it(self):
        import inspect

        from sglang.srt.managers import scheduler as S

        src = inspect.getsource(S)
        self.assertIn('refusal_detail=lambda: getattr(self, "_weg2_last_flush_refusal", "") or ""', src)
        fsrc = inspect.getsource(S.Scheduler.flush_cache)
        self.assertLess(fsrc.index('self._weg2_last_flush_refusal = ""'),
                        fsrc.index("self._weg2_last_flush_refusal = str(verdict_detail"))


# ---------------------------------------------------------------------------
# #1268 on the H77 harness, driven by the real Front.quiesce
# ---------------------------------------------------------------------------


class Protocol1268(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = H.FakeClock()
        p = mock.patch("time.monotonic", new=self.clock)
        p.start()
        self.addCleanup(p.stop)
        self.g = H.build_p_group(self.clock)
        pp0 = self.g.ranks[0]
        self.g.wrapper._refusal_detail = lambda: getattr(pp0, "_weg2_last_flush_refusal", "") or ""

    def pass_pp0(self):
        H.run_pass(self.g, 0)
        self.clock.advance(0.002)

    def lap_home(self):
        g = self.g
        while g.wire.inbox[1] or g.wire.inbox[2]:
            H.run_pass(g, 1)
            H.run_pass(g, 2)
        self.clock.advance(0.002)

    def _quiesce(self, script, fast=True):
        """``script[k]`` runs at the k-th front sleep (the group's time between polls)."""
        from sglang.srt.managers.io_struct import FlushCacheReqInput

        g = self.g
        bodies, sleeps = [], []

        async def rpc(_g, path, _body, _timeout):
            out = g.wrapper.handle(FlushCacheReqInput())
            body = "ok" if out.success else (out.message or "Flush cache failed.\n")
            bodies.append(body)
            return (200 if out.success else 400), body

        async def fake_sleep(s):
            sleeps.append(round(s, 4))
            step = script.get(len(sleeps))
            if step is not None:
                step()
            self.clock.advance(s)

        fake = SimpleNamespace(rpc=rpc, _health_inflight={})
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(fast), \
                mock.patch.object(front_mod.asyncio, "sleep", new=fake_sleep):
            ok = asyncio.run(front_mod.Front.quiesce(fake, SimpleNamespace(name="P")))
        return ok, bodies, sleeps

    def test_fast_re_polls_during_the_lap_keep_one_lap_and_one_flush(self):
        g = self.g
        stamped = {}

        def stamp():
            self.pass_pp0()
            stamped["ep"] = g.ranks[0]._weg2_vote_outstanding

        script = {1: stamp, 6: lambda: (self.lap_home(), self.pass_pp0())}
        with self.assertLogs(H.VOTE_LOGGER, level="INFO") as vlogs:
            ok, bodies, sleeps = self._quiesce(script)
        self.assertTrue(ok[0])
        self.assertEqual(sleeps, [0.001] * 6, "every refusal here was PENDING: re-polled at once")
        self.assertTrue(all(FW.PENDING_MARK in b for b in bodies[:-1]), bodies)
        self.assertEqual(len(bodies), 7, "the 200 on the first poll after the lap came home")
        self.assertEqual(int(g.ranks[0]._weg2_vote_epoch), int(stamped["ep"]), "a second lap was stamped")
        self.assertFalse([m for m in vlogs.output if "#1268 IDLE-ROUND stale" in m], vlogs.output)
        self.assertEqual(g.resets.n, 1, "exactly one group flush")

    def test_a_busy_rank_is_not_hammered_and_never_passes(self):
        g = self.g
        H.busy(g, [2], ["hicache_backup(4)"])
        script = {
            1: self.pass_pp0,                                    # stamp
            2: lambda: (self.lap_home(), self.pass_pp0()),       # NOT-IDLE lap home
            3: self.pass_pp0,                                    # stamp the re-wanted lap
            4: lambda: (H.idle(g, [2]), self.lap_home(), self.pass_pp0()),
        }
        ok, bodies, sleeps = self._quiesce(script)
        self.assertTrue(ok[0])
        self.assertEqual(g.resets.n, 1)
        busy_idx = [i for i, b in enumerate(bodies[:-1]) if FW.PENDING_MARK not in b]
        self.assertTrue(busy_idx, f"the NOT IDLE answer must not read as PENDING: {bodies}")
        for i in busy_idx:
            self.assertEqual(sleeps[i], 0.01, f"a busy group was re-polled at once: {sleeps} {bodies}")

    def test_switch_off_on_the_rank_keeps_the_old_cadence(self):
        with mock.patch.dict("os.environ", {FW.ENV_PENDING_MESSAGE: "0"}):
            script = {1: self.pass_pp0, 3: lambda: (self.lap_home(), self.pass_pp0())}
            ok, bodies, sleeps = self._quiesce(script)
        self.assertTrue(ok[0])
        self.assertTrue(all(s == 0.01 for s in sleeps), sleeps)
        self.assertEqual(self.g.resets.n, 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import unittest

    unittest.main()
