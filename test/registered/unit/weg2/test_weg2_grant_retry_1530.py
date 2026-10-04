# SPDX-License-Identifier: Apache-2.0
"""#1530 GRANT-RETRY throttle + GRANT-SHORT instrument (dual P, PP0).

B9g boot 5 (04.10. 20:10-20:22Z, deskq/done/1530-b9g-boot5-auswertung.md): PP0 retried every held
leg on EVERY scheduler pass. 'PP0 WAIT census' counted 253227 grants in 128 s (~2000/s), each one
opening every stage file and every card ledger; py-spy showed PP0 at 101 % CPU holding the GIL, the
forward/publish of the same process starved (Deadman HAENGT 2x ~90 s, 35x HTTP 503).

DANGER DIRECTIONS guarded here:
* default (SGLANG_WEG2_DUAL_GRANT_RETRY_MS=0) = every pass retries, exactly as before;
* a held rid is never starved: with the throttle on it still retries at least every N ms;
* a changed card ledger record lifts the throttle at once (a freed byte is granted without delay);
* the stage cache never serves a table a stage re-published;
* the first attempt of a rid is never throttled, a granted rid starts a fresh throttle;
* the GRANT-SHORT line is rate-limited by the WAIT backoff and never raises into the round.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _write(path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = [0] * 1000


class _Actor:
    page = 64
    _committed = 0
    ledger = None


class _Sched:
    class ps:
        pp_rank = 0
        pp_size = 2

    _weg2_store_held = {}


class ThrottleHelpers(CustomTestCase):
    def setUp(self):
        S._RETRY.clear()
        S._STAGE_CACHE.clear()
        self.d = tempfile.mkdtemp(prefix="g1530")
        self.led = [os.path.join(self.d, "wkv-%d" % i) for i in range(2)]
        for p in self.led:
            _write(p, b"\x01" * K._SIZE)
        self.stages = [{"ledger": p} for p in self.led]

    def test_default_is_off(self):
        self.assertEqual(S._retry_ms(), 0)

    def test_first_attempt_runs_then_throttled_until_n_ms(self):
        self.assertFalse(S._retry_throttled("r1", 50, 100.0))
        S._retry_note("r1", 100.0)
        self.assertTrue(S._retry_throttled("r1", 50, 100.049))
        self.assertFalse(S._retry_throttled("r1", 50, 100.051), "never starved past N ms")

    def test_n_calls_in_t_ms_give_at_most_t_over_n_attempts(self):
        attempts = 0
        for i in range(20000):                      # 2 s at 10 kHz = a spinning PP0
            now = 100.0 + i * 0.0001
            if not S._retry_throttled("r1", 50, now):
                attempts += 1
                S._retry_note("r1", now)
        self.assertLessEqual(attempts, 2000 // 50 + 1)
        self.assertGreaterEqual(attempts, 2000 // 50 - 1)

    def test_a_ledger_change_does_not_lift_the_throttle(self):
        # b9h: D's ledger changes on every tick; a signature check held the throttle open (~480/s)
        S._retry_note("r1", 100.0)
        self.assertTrue(S._retry_throttled("r1", 50, 100.001))
        _write(self.led[1], b"\x02" * K._SIZE)      # a card freed/committed bytes
        self.assertTrue(S._retry_throttled("r1", 50, 100.002))
        self.assertFalse(S._retry_throttled("r1", 50, 100.051), "only the interval lifts it")

    def test_a_constantly_changing_ledger_still_gives_at_most_t_over_n_attempts(self):
        attempts = 0
        for i in range(20000):                      # 2 s at 10 kHz, the ledger rewritten every pass
            now = 100.0 + i * 0.0001
            _write(self.led[0], bytes([i % 251]) * K._SIZE)
            if not S._retry_throttled("r1", 50, now):
                attempts += 1
                S._retry_note("r1", now)
        self.assertLessEqual(attempts, 2000 // 50 + 1)
        self.assertGreaterEqual(attempts, 2000 // 50 - 1)

    def test_unreadable_ledger_does_not_matter(self):
        S._retry_note("r1", 100.0)
        self.assertTrue(S._retry_throttled("r1", 50, 100.001))   # the ledger plays no part

    def test_rids_are_independent(self):
        S._retry_note("r1", 100.0)
        self.assertFalse(S._retry_throttled("r2", 50, 100.001))

    def test_a_granted_rid_starts_a_fresh_throttle(self):
        S._retry_note("r1", 100.0)
        S._wait_granted("r1")
        self.assertFalse(S._retry_throttled("r1", 50, 100.001))

    def test_stage_cache_parses_once_and_invalidates_on_republish(self):
        p = os.path.join(self.d, "stage0.json")
        _write(p, json.dumps({"step": 1, "v": 1}).encode())
        with mock.patch("json.load", wraps=json.load) as jl:
            a = S._load_stage(p, True)
            b = S._load_stage(p, True)
            self.assertEqual(jl.call_count, 1, "second read must come from the cache")
        self.assertIs(a, b)
        tmp = p + ".tmp"                             # publish_stage: tmp + os.replace
        _write(tmp, json.dumps({"step": 1, "v": 2, "pad": "x" * 10}).encode())
        os.replace(tmp, p)
        self.assertEqual(S._load_stage(p, True)["v"], 2, "a re-published table must not be served stale")

    def test_uncached_load_is_the_old_behaviour(self):
        p = os.path.join(self.d, "stage1.json")
        _write(p, json.dumps({"v": 1}).encode())
        with mock.patch("json.load", wraps=json.load) as jl:
            S._load_stage(p, False)
            S._load_stage(p, False)
            self.assertEqual(jl.call_count, 2)
        self.assertFalse(S._STAGE_CACHE)


class Pp0GrantThrottle(CustomTestCase):
    """Through pp0_grant itself, with the card ledgers and the group grant faked."""

    def setUp(self):
        S._RETRY.clear()
        S._STAGE_CACHE.clear()
        S._reset_wait_log()
        self.d = tempfile.mkdtemp(prefix="g1530p")
        self.led = [os.path.join(self.d, "wkv-%d" % i) for i in range(2)]
        for p in self.led:
            _write(p, b"\x00" * K._SIZE)
        self.paths = [os.path.join(self.d, "wkvs-pp%d.json" % i) for i in range(2)]
        for i, p in enumerate(self.paths):
            _write(p, json.dumps({"ledger": self.led[i], "step": 4096, "top": 196608,
                                  "bytes": [0] + [i + 1] * 64}).encode())
        self.calls = 0
        self.clock = [1000.0]

    def _run(self, ms, n=2000, step_s=0.0005, rid="weg2-0-1"):
        def gg(*a, **k):
            self.calls += 1
            return 0
        patches = [
            mock.patch.object(S, "_actor", lambda s: _Actor()),
            mock.patch.object(S, "stage_file", lambda tag, r, root="/dev/shm": self.paths[r]),
            mock.patch.object(S, "_retry_ms", lambda: ms),
            mock.patch.object(S, "_dual_layout_env", lambda: True),
            mock.patch.object(S, "return_untold_grant", lambda *a, **k: None),
            mock.patch.object(S, "live_grant_tokens", lambda *a, **k: 0),
            mock.patch.object(S, "_older_waits", lambda *a, **k: []),
            mock.patch.object(S, "group_grant", gg),
            mock.patch.object(S, "_now", lambda: self.clock[0]),
            mock.patch("sglang.srt.weg2.dual_grant_wait.grant_held_by_d", lambda *a, **k: None),
        ]
        for p in patches:
            p.start()
        try:
            req, sched = _Req(rid), _Sched()
            for i in range(n):
                self.clock[0] = 1000.0 + i * step_s
                self.assertEqual(S.pp0_grant(sched, req), 0)
        finally:
            for p in patches:
                p.stop()

    def test_off_every_pass_retries(self):
        self._run(0, n=500)
        self.assertEqual(self.calls, 500)

    def test_on_a_spinning_pp0_makes_at_most_t_over_n_attempts(self):
        self._run(50, n=2000, step_s=0.0005)         # 1.0 s of passes
        self.assertLessEqual(self.calls, 1000 // 50 + 1)
        self.assertGreaterEqual(self.calls, 1000 // 50 - 1)

    def test_on_a_ledger_change_between_passes_still_waits_the_interval(self):
        def gg(*a, **k):
            self.calls += 1
            return 0
        with mock.patch.object(S, "_actor", lambda s: _Actor()), \
                mock.patch.object(S, "stage_file", lambda tag, r, root="/dev/shm": self.paths[r]), \
                mock.patch.object(S, "_retry_ms", lambda: 1000), \
                mock.patch.object(S, "_dual_layout_env", lambda: True), \
                mock.patch.object(S, "return_untold_grant", lambda *a, **k: None), \
                mock.patch.object(S, "live_grant_tokens", lambda *a, **k: 0), \
                mock.patch.object(S, "_older_waits", lambda *a, **k: []), \
                mock.patch.object(S, "group_grant", gg), \
                mock.patch.object(S, "_now", lambda: self.clock[0]), \
                mock.patch("sglang.srt.weg2.dual_grant_wait.grant_held_by_d", lambda *a, **k: None):
            req, sched = _Req("weg2-0-2"), _Sched()
            for _ in range(5):
                S.pp0_grant(sched, req)
            self.assertEqual(self.calls, 1)
            _write(self.led[0], b"\x07" * K._SIZE)
            S.pp0_grant(sched, req)
            self.assertEqual(self.calls, 1, "a ledger change must not lift the throttle")
            self.clock[0] += 1.001
            S.pp0_grant(sched, req)
            self.assertEqual(self.calls, 2)


class GrantShortLine(CustomTestCase):
    def setUp(self):
        S._reset_wait_log()

    def test_detail_is_on_the_backoff_curve_and_never_raises(self):
        lines = []
        clock = [1000.0]
        boom = {"n": 0}

        def detail():
            boom["n"] += 1
            raise RuntimeError("detail exploded")
        with mock.patch.object(S, "_now", lambda: clock[0]), \
                mock.patch.object(S.logger, "info", lambda msg, *a: lines.append(msg % a)):
            for i in range(20000):                   # 200 s at 100 retries/s
                clock[0] = 1000.0 + i * 0.01
                S._log_wait("weg2-0-9", 62343, detail)
        short = [l for l in lines if "#1530 GRANT-SHORT" in l]
        self.assertEqual(boom["n"], len([l for l in lines if "PP0 WAIT rid=weg2-0-9" in l]))
        self.assertLessEqual(boom["n"], 9)
        self.assertEqual(short, [], "a failing detail adds no line and raises nothing")

    def test_detail_names_every_card(self):
        d = tempfile.mkdtemp(prefix="g1530s")
        paths = []
        for i, free in enumerate((600, 1700)):
            p = os.path.join(d, "wkv-%d" % i)
            led = K.CardKvLedger(p, "D")
            led.contribute(free)
            paths.append(p)
        stages = [{"ledger": paths[0], "step": 4096, "top": 8192, "bytes": [0, 1000, 2000]},
                  {"ledger": paths[1], "step": 4096, "top": 8192, "bytes": [0, 100, 200]}]
        txt = S._grant_short_detail(stages, {0: 100}, 8192, 20000, 3, "pressure_on_P=0 d_demand=1")
        self.assertIn("level=8192 sum=20000 older=3 hold=pressure_on_P=0 d_demand=1", txt)
        self.assertIn("0:need=1900,have=600", txt)
        self.assertIn("1:need=200,have=1700", txt)

    def test_detail_failure_is_a_note(self):
        self.assertIn("detail failed", S._grant_short_detail([], {}, 1, 1, 0, None))


if __name__ == "__main__":
    import unittest

    unittest.main()
