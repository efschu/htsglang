# SPDX-License-Identifier: Apache-2.0
"""NF W98 HOST-WARN (12.10.2026): host-RAM guards WARN, they do not stop the server.

USER RULE 06.10. (verbatim in /root/.claude/CLAUDE.md, "Host-RAM"): "Es gilt KEINE
Verweigerung wegen Host-RAM; Ledger, Launcher und Startskripte warnen nur ... Man
laesst es darauf ankommen, ob es OOM geht. VRAM-Verweigerungen bleiben
unberuehrt."

MEASURED DEATH (boot dkrnfint4h6ablxcw3spmlrhlvwbar1dauer10121858, int24g, front log
22:37:44,498Z): ``W98 PdFlipHostRateLatched: cushion=0.38 GiB BELOW the floor 1.50
while shmem is STILL RISING (now=89.78 GiB, shmem=61.27, headroom=6.12 GiB <=
relevance 32.00, gaps_seen=0)`` -> ``PDFLIP STOP IN-FLIGHT`` (6 requests 503) ->
``STOP STATE lifecycle=serving stop_request=True`` -> ``PDFLIP-FLIP REFUSED-STOP
epoch=154``; /health 503. The server stopped itself on a host-RAM reading.

What is pinned here, in the direction that was red on c59d8bda8d:

  * the REPRO: the front's guard loop, driven with exactly that reading, does
    NOT call ``do_stop`` and the front is not in STOP (red on the base: the old
    loop called ``do_stop("W98 PdFlipHostRateLatched")`` on the first tick pair);
  * the warning carries the full numbers, once per episode (no log storm), and
    re-arms when the cushion is back at the floor;
  * W22 (level) and W100 (sleep-leg cushion) warn too;
  * the launch guard no longer raises ``PdFlipLaunchHostLatched``;
  * R1 -- what must NOT change: every VRAM / rank-death / drain / store-identity
    ``do_stop`` site is still there, ``dormant_residue_over`` (W19, VRAM) still
    names an overflowing card, and the pure host arithmetic of the old verdict is
    byte-identical for every caller that does not ask for ``warn_only``.

Hermetic: no GPU, no HTTP, no cgroup reads (every reading is injected).
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import os
import tempfile
import time as real_time
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as front_mod
from flliper.srt.pdflip import host_ledger as hl
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: the marker the warnings carry; spelled out here, not imported, so this file
#: also RUNS on the base (where the constant does not exist) and goes red there
#: on the behaviour, not on an ImportError.
MARKER = "HOST-RAM-WARN"

#: int24g 22:37:44Z, the reading that stopped the server: cushion = file - shmem.
INT24G = dict(nonreclaim_gib=89.78, shmem_gib=61.27, cushion=0.38, memfree_gib=1.0)


def _pr(nonreclaim, shmem, cushion, memfree=1.0):
    return {
        "current_gib": nonreclaim, "nonreclaim_gib": nonreclaim,
        "file_reclaimable_gib": 0.1, "anon_gib": 20.0, "shmem_gib": shmem,
        "file_gib": shmem + cushion, "memfree_gib": memfree,
        "memavail_gib": memfree, "max_gib": None, "source": "test",
    }


def _front():
    return front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="w98warn",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1,
    )


class _Cancel(Exception):
    pass


def _drive_sampler(f, readings, step_s=0.5, t0=1_000_000.0):
    """Run ``Front.host_watermark_sampler`` over ``readings`` (one per tick).

    Returns ``(stop_calls, log_records)``. The clock advances ``step_s`` per
    tick, the guard's own sleep is the tick, the loop ends after the last
    reading. ``do_stop`` is replaced by a recorder that also performs the one
    thing a real stop does to the state (``state = STOP``), so "not in STOP" is
    a measured fact of the run and not an assumption.
    """
    stop_calls = []
    clock = {"t": t0, "i": 0}

    def fake_time():
        return clock["t"]

    def fake_pressure():
        i = min(clock["i"], len(readings) - 1)
        return dict(readings[i])

    async def fake_sleep(_s):
        clock["i"] += 1
        clock["t"] += step_s
        if clock["i"] >= len(readings):
            raise _Cancel()

    def fake_do_stop(name, detail):
        stop_calls.append(name)
        f.state = "STOP"

    fake_front_time = types.SimpleNamespace(
        time=fake_time, monotonic=real_time.monotonic,
        perf_counter=real_time.perf_counter, sleep=real_time.sleep,
        strftime=real_time.strftime, gmtime=real_time.gmtime)

    async def run():
        try:
            await f.host_watermark_sampler()
        except _Cancel:
            pass

    with mock.patch.object(hl, "read_cgroup_pressure", fake_pressure), \
         mock.patch.object(hl, "read_cgroup", lambda *a, **k: {"current": 1}), \
         mock.patch.object(hl, "read_cgroup_anon_bytes", lambda *a, **k: None), \
         mock.patch.object(hl, "arena_fill_gib", lambda *a, **k: None), \
         mock.patch.object(front_mod, "time", fake_front_time), \
         mock.patch.object(front_mod.asyncio, "sleep", fake_sleep), \
         mock.patch.object(f, "do_stop", fake_do_stop):
        with mock.patch.object(front_mod.logger, "warning") as w, \
             mock.patch.object(front_mod.logger, "info") as i, \
             mock.patch.object(front_mod.logger, "error") as e:
            asyncio.run(run())
    lines = {
        "warning": [str(c.args[0]) % c.args[1:] if len(c.args) > 1 else str(c.args[0]) for c in w.call_args_list],
        "info": [str(c.args[0]) % c.args[1:] if len(c.args) > 1 else str(c.args[0]) for c in i.call_args_list],
        "error": [str(c.args[0]) % c.args[1:] if len(c.args) > 1 else str(c.args[0]) for c in e.call_args_list],
    }
    return stop_calls, lines


def _int24g_series(n_after=40):
    """A calm tick, then the int24g reading with shmem rising, then it HOLDS."""
    calm = _pr(70.0, 40.0, 6.0, memfree=20.0)
    # the tick before: same pool, cushion still healthy, shmem a little lower
    prev = _pr(89.50, 61.00, 6.0, memfree=20.0)
    bad = _pr(INT24G["nonreclaim_gib"], INT24G["shmem_gib"], INT24G["cushion"])
    return [calm, calm, prev, bad] + [bad] * n_after


class TheInt24gDeathIsAWarningNow(CustomTestCase):
    """THE REPRO. Red on c59d8bda8d (``do_stop`` on the first W98 verdict)."""

    def test_the_int24g_reading_does_not_stop_the_front(self):
        f = _front()
        stop_calls, lines = _drive_sampler(f, _int24g_series())
        self.assertEqual(stop_calls, [], f"the front stopped itself on host RAM: {stop_calls}")
        self.assertNotEqual(f.state, "STOP")

    def test_it_warns_once_with_the_full_numbers(self):
        f = _front()
        stop_calls, lines = _drive_sampler(f, _int24g_series(n_after=200))
        warns = [l for l in lines["warning"] if MARKER in l and " W98 " in l]
        self.assertEqual(len(warns), 1, warns)
        w = warns[0]
        for token in ("cushion=0.38", "1.50", "now=89.78", "shmem=61.27",
                      "headroom=6.12", "relevance 32.00", "WARNING ONLY"):
            self.assertIn(token, w, f"the warning does not carry {token}")
        # the old verdict words are gone from the log: death watchers and greps
        # read those as "the front stopped"
        for l in lines["warning"] + lines["error"] + lines["info"]:
            self.assertNotIn("PdFlipHostRateLatched", l)
            self.assertNotIn("PDFLIP STOP", l)
        self.assertEqual(f.counters["host_rate_latch"], 1)

    def test_no_request_is_failed(self):
        """No STOP -> no ``STOP IN-FLIGHT`` 503 sweep, state untouched."""
        f = _front()
        _drive_sampler(f, _int24g_series())
        self.assertEqual(f.counters["host_rate_latch"], 1)
        self.assertIsNone(getattr(f, "stop", None))

    def test_a_flapping_cushion_is_not_a_log_storm(self):
        """below/above the floor every few ticks: re-armed, but bounded."""
        f = _front()
        calm = _pr(70.0, 40.0, 6.0, memfree=20.0)
        series = [calm, _pr(89.50, 61.00, 0.38)]
        shm = 61.27
        for _ in range(40):        # 40 flaps, 4 ticks each = 80 s at 0.5 s/tick
            # shmem keeps RISING through every flap (that is the latch's gate)
            series += [_pr(89.78, shm + 0.05, 0.38), _pr(89.78, shm + 0.10, 0.38),
                       _pr(89.78, shm + 0.15, 3.0), _pr(89.78, shm + 0.20, 3.0)]
            shm += 0.20
        stop_calls, lines = _drive_sampler(f, series)
        warns = [l for l in lines["warning"] if MARKER in l and " W98 " in l and "CLEARED" not in l]
        # 80 s of flapping at a 30 s minimum interval: a handful of lines, not 40
        self.assertEqual(stop_calls, [])
        self.assertLessEqual(len(warns), 4, len(warns))
        self.assertGreaterEqual(len(warns), 2)

    def test_the_episode_ends_on_the_measurement_and_rearms(self):
        f = _front()
        calm = _pr(70.0, 40.0, 6.0, memfree=20.0)
        bad_a = _pr(89.50, 61.00, 0.38)
        bad_b = _pr(89.78, 61.27, 0.38)
        ok = _pr(89.78, 61.27, 3.0)
        # 70 s later (140 ticks at 0.5 s) the second episode is past the interval
        series = [calm, bad_a, bad_b, bad_b, ok] + [ok] * 140 + [_pr(89.9, 61.40, 0.30), _pr(90.0, 61.50, 0.30), _pr(90.0, 61.50, 0.30)]
        stop_calls, lines = _drive_sampler(f, series)
        self.assertEqual(stop_calls, [])
        warns = [l for l in lines["warning"] if MARKER in l and " W98 " in l and "CLEARED" not in l]
        cleared = [l for l in lines["info"] if MARKER in l and "W98 CLEARED" in l]
        self.assertEqual(len(warns), 2, warns)
        self.assertEqual(len(cleared), 1, cleared)


class TheLevelGuardWarnsToo(CustomTestCase):
    """W22 (non-reclaimable level over the runtime bound): warn, bounded, clears."""

    def _over(self):
        m = hl.resolve_margin(drift_mib_per_min=None)
        bound = hl.reap_mark_gib() - m.runtime_total_gib
        # cushion 6 GiB (above the floor): the W98 latch stays out of it
        return _pr(bound + 1.0, 40.0, 6.0, memfree=20.0), _pr(bound - 5.0, 40.0, 6.0, memfree=20.0)

    def test_over_the_bound_warns_and_does_not_stop(self):
        over, under = self._over()
        f = _front()
        # one level check per host_watermark_period_s (15 s) = 30 ticks at 0.5 s
        series = [under] * 2 + [over] * 200
        stop_calls, lines = _drive_sampler(f, series)
        self.assertEqual(stop_calls, [])
        warns = [l for l in lines["warning"] if MARKER in l and " W22 " in l]
        self.assertGreaterEqual(len(warns), 1)
        # 100 s over the bound with a 300 s reminder: exactly the enter line
        self.assertEqual(len(warns), 1, warns)
        self.assertIn("WARNING ONLY", warns[0])
        self.assertNotIn("PdFlipHostWatermarkBreached", warns[0])
        self.assertEqual(f.counters["host_watermark_breach"], 1)

    def test_it_clears_and_rearms(self):
        over, under = self._over()
        f = _front()
        series = [under] * 2 + [over] * 70 + [under] * 70
        stop_calls, lines = _drive_sampler(f, series)
        self.assertEqual(stop_calls, [])
        self.assertEqual(len([l for l in lines["info"] if MARKER in l and "W22 CLEARED" in l]), 1)


class TheSleepLegGateWarns(CustomTestCase):
    """W100: the cushion deficit before a sleep leg is a warning, the leg runs."""

    def _front_with_record(self, td, delta):
        import json

        path = os.path.join(td, "rec.json")
        with open(path, "w") as fh:
            json.dump({"samples": [{
                "group": "D", "boot_tag": "weg2xsn25", "at": "2026-09-13T07:00Z",
                "shmem_delta_gib": delta, "rss_shmem_gib": 48.0}]}, fh)
        f = front_mod.Front.__new__(front_mod.Front)
        f.measured_record = path
        f.counters = __import__("collections").Counter()
        return f

    def test_the_xsn25_numbers_no_longer_refuse_the_leg(self):
        with tempfile.TemporaryDirectory() as td:
            f = self._front_with_record(td, 4.32)
            with mock.patch.object(hl, "read_cgroup_pressure",
                                   return_value={"file_gib": 62.697, "shmem_gib": 60.0}), \
                 mock.patch.object(front_mod.logger, "warning") as w:
                self.assertIsNone(f._sleep_leg_gate("D"))     # old: raised PdFlipSleepLegCushionDeficit
                self.assertIsNone(f._sleep_leg_gate("D"))     # second call: counted, not repeated
            msgs = [str(c.args[0]) % c.args[1:] for c in w.call_args_list]
            self.assertEqual(len(msgs), 1, msgs)
            for token in (MARKER, "W100", "need_gib=4.32", "cushion_gib=2.70", "WARNING ONLY"):
                self.assertIn(token, msgs[0])
            self.assertEqual(f.counters["host_sleep_leg_warn"], 2)

    def test_the_pure_function_still_computes_the_verdict(self):
        with self.assertRaises(hl.PdFlipSleepLegCushionDeficit):
            hl.refuse_sleep_leg_deficit(2.697, 4.32, source="record:weg2xsn25", group="D")


class TheLaunchGuardWarns(CustomTestCase):
    def test_no_launch_refusal_remains_for_the_latch(self):
        from flliper.srt.pdflip import launcher as lc

        src = inspect.getsource(lc)
        hand = src[src.index("_launch_guard.stop("):src.index("front_argv = front_argv_for(")]
        self.assertNotIn("raise", hand)
        self.assertNotIn("PdFlipLaunchHostLatched", hand)
        g = src[src.index("class LaunchGuard"):src.index("def served_model_name")]
        self.assertIn("warn_only=True", g)
        self.assertNotIn("self.verdict = line", g)

    def test_the_guard_thread_logs_and_never_records_a_verdict(self):
        from flliper.srt.pdflip import launcher as lc

        logged = []
        g = lc.LaunchGuard(logged.append, reap_mark_gib=95.90, period_s=0.01)
        seq = [_pr(89.50, 61.00, 0.38), _pr(89.78, 61.27, 0.38)]
        n = {"i": 0}

        def fake_pressure():
            r = seq[min(n["i"], len(seq) - 1)]
            n["i"] += 1
            if n["i"] > 6:
                g._stop.set()
            return dict(r)

        with mock.patch.object(hl, "read_cgroup_pressure", fake_pressure):
            g._run()
        self.assertIsNone(g.verdict)
        self.assertEqual(g.warns, 1)
        self.assertEqual(len([l for l in logged if MARKER in l and " W98 " in l]), 1, logged)
        self.assertIsNone(g.stop("test"))


class TheLatchArithmeticIsUnchangedForOtherCallers(CustomTestCase):
    """R1: the pure host arithmetic is byte-identical without ``warn_only``."""

    def test_default_latch_still_returns_the_old_verdict(self):
        lat = hl.RateLatch(reap_mark_gib=95.90)
        lat.observe(0.0, 89.50, cushion_gib=0.38, shmem_gib=61.00, free_gib=1.0)
        line = lat.observe(0.5, 89.78, cushion_gib=0.38, shmem_gib=61.27, free_gib=1.0)
        self.assertTrue(line.startswith("W98 PdFlipHostRateLatched: cushion=0.38 GiB BELOW"), line)
        self.assertTrue(lat.latched)
        self.assertIsNone(lat.observe(1.0, 89.9, cushion_gib=0.30, shmem_gib=61.5, free_gib=1.0))
        self.assertTrue(lat.latched)       # one-shot, never re-armed by itself

    def test_default_watermark_verdict_keeps_its_words(self):
        m = hl.resolve_margin(drift_mib_per_min=None)
        over = (hl.reap_mark_gib() - m.runtime_total_gib + 2.0)
        line = hl.watermark_breach_verdict(int(over * hl.GIB), margin=m, nonreclaim_gib=over)
        self.assertTrue(line.startswith("W22 PdFlipHostWatermarkBreached current="), line)
        self.assertIn("CONTROLLED TEARDOWN", line)
        warn = hl.watermark_breach_verdict(int(over * hl.GIB), margin=m, nonreclaim_gib=over, warn_only=True)
        self.assertTrue(warn.startswith(MARKER + " W22 "), warn)
        self.assertNotIn("CONTROLLED TEARDOWN", warn.upper())

    def test_the_warn_latch_writes_the_same_numbers_as_the_stop_latch(self):
        a = hl.RateLatch(reap_mark_gib=95.90)
        b = hl.RateLatch(reap_mark_gib=95.90, warn_only=True)
        for lat in (a, b):
            lat.observe(0.0, 89.50, cushion_gib=0.38, shmem_gib=61.00, free_gib=1.0)
        la = a.observe(0.5, 89.78, cushion_gib=0.38, shmem_gib=61.27, free_gib=1.0)
        lb = b.observe(0.5, 89.78, cushion_gib=0.38, shmem_gib=61.27, free_gib=1.0)
        self.assertIsNotNone(la)
        self.assertIsNotNone(lb)
        for token in ("cushion=0.38", "now=89.78", "shmem=61.27", "headroom=6.12"):
            self.assertIn(token, la)
            self.assertIn(token, lb)


class TheVramAndRankDeathStopsAreStillStops(CustomTestCase):
    """R1 / the danger direction: only the two host-RAM guards lost their stop."""

    def _do_stop_names(self):
        src = inspect.getsource(front_mod)
        names = []
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "do_stop" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                names.append(node.args[0].value)
        return names

    def test_no_host_ram_literal_reaches_do_stop(self):
        names = self._do_stop_names()
        for n in names:
            self.assertNotIn("HostRateLatched", n)
            self.assertNotIn("HostWatermarkBreached", n)

    def test_the_vram_rankdeath_drain_and_identity_stops_remain(self):
        names = self._do_stop_names()
        # W17 (dead rank / group, a TOTAL of two sites), W19 (VRAM residue per
        # card), W4 (VRAM state undefined after a failed leg), W2/W3 (drain), W9
        for want, at_least in (("W17 PdFlipGroupDead", 2), ("W19 DormantResidueRefused", 1),
                               ("W4 PdFlipWakeRefused", 3), ("W2 PdFlipDrainStuck", 1),
                               ("W3 PdFlipDrainWitnessUnreachable", 1), ("W9 PdFlipStoreIdentityMismatch", 1)):
            self.assertGreaterEqual(names.count(want), at_least, (want, names.count(want)))

    def test_a_vram_overflow_is_still_named_by_the_w19_check(self):
        over = front_mod.dormant_residue_over({"GPU-a": 3000, "GPU-b": 100},
                                              {"GPU-a": 2000, "GPU-b": 2000}, False)
        self.assertEqual(over, {"GPU-a": (3000, 2000)})

    def test_a_named_stop_still_fails_the_queue_and_sets_stop(self):
        """``do_stop`` itself is untouched: a VRAM-class name still ends in STOP."""
        f = _front()
        with mock.patch.object(front_mod.logger, "error"):
            f.do_stop("W19 DormantResidueRefused", "measured D_c exceeds the reserve")
        self.assertEqual(f.state, "STOP")

    def test_the_guard_loop_has_no_do_stop_call(self):
        src = inspect.getsource(front_mod.Front.host_watermark_sampler)
        self.assertNotIn("self.do_stop(", src)
        self.assertIn("warn_only=True", src)


if __name__ == "__main__":
    unittest.main()
