# SPDX-License-Identifier: Apache-2.0
"""#1640 GRANT-INFEASIBLE marker + GRANT_INFEASIBLE_SKIP (dual P, PP0).

Boots b9j/b9k/b9l (deskq/done/1640-live-floor-design.md): card 0 of the 27B dual has a ~3.1 GB pool
but level 196608 needs 4.08 GB there. Such a grant is short even if D gave back EVERYTHING, so only a
falling level frees it. As the OLDEST waiter it then blocks every younger grant after
SGLANG_WEG2_DUAL_BYPASS_HEAD_AGE_S (hold=older-head -> '4 queued, 0 running' wedges of 62-65 s).

DANGER DIRECTIONS guarded here:
* default (SGLANG_WEG2_DUAL_GRANT_INFEASIBLE_SKIP off): ``_older_waits`` and the grant path are the
  old ones, nothing is evaluated per attempt, no flag is ever set;
* the marker fires ONLY when need > ledger free + D committed on some card (a merely short grant
  that D could free is NOT infeasible), is rate-limited by the WAIT backoff and never raises;
* switch on: an infeasible head no longer holds younger grants back, a FEASIBLE older head still does;
* the head keeps retrying itself (the flag is refreshed on every attempt, cleared on grant);
* PP0-local: a non-PP0 rank neither evaluates nor sets anything (no collective, no wall clock in the
  decision itself -- only the existing monotonic age of the head).
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


def _ledger(path, budget: int, d_committed: int, p_committed: int = 0):
    """A card ledger with pool ``budget``, D holding ``d_committed`` and P ``p_committed``."""
    K.CardKvLedger(path, "D").contribute(budget, committed=d_committed)
    if p_committed:
        pl = K.CardKvLedger(path, "P")
        pl.join(budget)
        got, _ = pl.request(p_committed)
        assert got == p_committed


def _stage(path_ledger, bytes_level):
    # step 4096, top 196608: tokens 1064 -> level 4096 -> index 1
    return {"ledger": path_ledger, "step": 4096, "top": 196608, "bytes": [0] + [bytes_level] * 64}


class _Req:
    def __init__(self, rid, wait=False):
        self.rid = rid
        self.origin_input_ids = [0] * 1000
        self._dual_kv_wait = wait


class _Actor:
    page = 64
    _committed = 0
    ledger = None

    def map_granted(self, lvl, charged=0):      # the grant that finally fits maps PP0's own card
        self.mapped = (lvl, charged)


class _Sched:
    def __init__(self, pp_rank=0, held=None):
        self.ps = type("ps", (), {"pp_rank": pp_rank, "pp_size": 2})()
        self._weg2_store_held = held or {}


class InfeasibleCards(CustomTestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="g1640")
        self.led = [os.path.join(self.d, "wkv-%d" % i) for i in range(2)]

    def test_feasible_if_d_could_free_it(self):
        _ledger(self.led[0], 3100, d_committed=2000)        # free 1100, need 2500 <= 1100 + 2000
        _ledger(self.led[1], 6000, d_committed=100)
        st = [_stage(self.led[0], 2500), _stage(self.led[1], 500)]
        self.assertEqual(S.infeasible_cards(st, {}, 1064), [], "short, but D could give it back: NOT infeasible")

    def test_infeasible_even_with_d_at_zero_names_the_card(self):
        _ledger(self.led[0], 3100, d_committed=2000)        # free 1100, pool 3100, need 4080 > 3100
        _ledger(self.led[1], 6000, d_committed=100)
        st = [_stage(self.led[0], 4080), _stage(self.led[1], 500)]
        out = S.infeasible_cards(st, {}, 1064)
        self.assertEqual(out, [(0, 4080, 1100, 0, 2000)])

    def test_p_committed_is_not_counted_as_available(self):
        _ledger(self.led[0], 3100, d_committed=500, p_committed=2000)   # free 600, D 500, P 2000
        st = [_stage(self.led[0], 3000), _stage(self.led[1], 0)]
        _ledger(self.led[1], 100, d_committed=0)
        # need 3000 > free 600 + D 500: P's own 2000 is held by running requests -> infeasible NOW
        self.assertEqual([c[0] for c in S.infeasible_cards(st, {}, 1064)], [0])

    def test_covered_mapping_lowers_the_need(self):
        _ledger(self.led[0], 3100, d_committed=0, p_committed=2300)   # free 800
        _ledger(self.led[1], 6000, d_committed=0)
        st = [_stage(self.led[0], 3200), _stage(self.led[1], 100)]
        self.assertEqual([c[0] for c in S.infeasible_cards(st, {}, 1064)], [0], "uncovered: 3200 > 800")
        self.assertEqual([c[0] for c in S.infeasible_cards(st, {0: 2300}, 1064)], [0], "need 900 > free 800 + D 0")
        self.assertEqual(S.infeasible_cards(st, {0: 2500}, 1064), [], "PP0 maps 2500: need 700 <= free 800")

    def test_missing_ledger_and_garbage_never_raise(self):
        st = [_stage(os.path.join(self.d, "absent"), 99999999)]
        self.assertEqual(S.infeasible_cards(st, {}, 1064), [])
        self.assertEqual(S.infeasible_cards([], {}, 1064), [])
        self.assertEqual(S.infeasible_cards([{"nope": 1}], {}, 1064), [])


class MarkerOnTheBackoffCurve(CustomTestCase):
    def setUp(self):
        S._reset_wait_log()

    def _run(self, infeasible, n=20000, dt=0.01):
        lines, clock = [], [1000.0]
        with mock.patch.object(S, "_now", lambda: clock[0]), \
                mock.patch.object(S.logger, "info", lambda msg, *a: lines.append(msg % a)):
            for i in range(n):
                clock[0] = 1000.0 + i * dt
                S._log_wait("weg2-0-9", 62343, None, infeasible)
        return lines

    def test_fires_only_when_infeasible_and_is_rate_limited(self):
        lines = self._run(lambda: S._infeasible_line("weg2-0-9", 62343, 196608, [(0, 4080, 1100, 0, 2000)]))
        inf = [l for l in lines if "#1640 GRANT-INFEASIBLE" in l]
        self.assertGreaterEqual(len(inf), 5)
        self.assertLessEqual(len(inf), 9, "200 s of waits at 100/s stay on the doubling backoff")
        self.assertIn("0:need=4080,pool=3100,have=1100,P=0,D=2000", inf[0])
        self.assertIn("level_tokens=196608", inf[0])

    def test_feasible_writes_nothing(self):
        lines = self._run(lambda: None, n=3000)
        self.assertEqual([l for l in lines if "GRANT-INFEASIBLE" in l], [])

    def test_a_failing_marker_raises_nothing_and_adds_no_line(self):
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise RuntimeError("marker exploded")
        lines = self._run(boom, n=3000)
        self.assertGreater(calls["n"], 0)
        self.assertEqual([l for l in lines if "GRANT-INFEASIBLE" in l], [])
        self.assertTrue([l for l in lines if "PP0 WAIT rid=weg2-0-9" in l], "the WAIT line itself stays")


class Pp0GrantThroughTheGate(CustomTestCase):
    """pp0_grant itself, with REAL card ledgers + real group_grant; card 0 pool 3100, level needs 4080."""

    def setUp(self):
        S._RETRY.clear()
        S._STAGE_CACHE.clear()
        S._reset_wait_log()
        self.d = tempfile.mkdtemp(prefix="g1640p")
        self.led = [os.path.join(self.d, "wkv-%d" % i) for i in range(2)]
        _ledger(self.led[0], 3100, d_committed=2000)
        _ledger(self.led[1], 6000, d_committed=100)
        self.paths = [os.path.join(self.d, "wkvs-pp%d.json" % i) for i in range(2)]
        for p, st in zip(self.paths, (_stage(self.led[0], 4080), _stage(self.led[1], 500))):
            with open(p, "w") as f:
                json.dump(st, f)
        self.clock = [1000.0]
        self.lines = []

    def _grant(self, rid, skip, sched, *, rank=0):
        patches = [
            mock.patch.object(S, "_actor", lambda s: _Actor()),
            mock.patch.object(S, "stage_file", lambda tag, r, root="/dev/shm": self.paths[r]),
            mock.patch.object(S, "_dual_layout_env", lambda: True),
            mock.patch.object(S, "_retry_ms", lambda: 0),
            mock.patch.object(S, "_infeasible_skip_armed", lambda: skip),
            mock.patch.object(S, "return_untold_grant", lambda *a, **k: None),
            mock.patch.object(S, "live_grant_tokens", lambda *a, **k: 0),
            mock.patch.object(S, "_now", lambda: self.clock[0]),
            mock.patch.object(S.logger, "info", lambda msg, *a: self.lines.append(msg % a)),
            mock.patch("sglang.srt.weg2.dual_grant_wait.grant_held_by_d", lambda *a, **k: None),
        ]
        for p in patches:
            p.start()
        try:
            with mock.patch.object(S, "group_grant", wraps=S.group_grant) as gg:
                r = S.pp0_grant(sched, _Req(rid))
            return r, gg.call_count
        finally:
            for p in patches:
                p.stop()

    def _head(self):
        """An old, held head: waiting 100 s (> the 60 s head age), flagged infeasible by its own retries."""
        head = _Req("weg2-0-9", wait=True)
        S._WAITS["weg2-0-9"] = [self.clock[0] - 100.0, self.clock[0], 1.0, 5, 0]
        return head, _Sched(held={"x": head})

    def test_default_off_old_behaviour_no_flag_no_evaluation(self):
        head, sched = self._head()
        S._INFEASIBLE["weg2-0-9"] = True                       # even a stale flag changes nothing when off
        r, ggn = self._grant("weg2-0-10", False, sched)
        self.assertEqual((r, ggn), (0, 0), "the infeasible head still holds the younger grant: hold=older-head")
        self.assertTrue([l for l in self.lines if "hold=older-head" in l])
        self.assertEqual(set(S._INFEASIBLE), {"weg2-0-9"}, "no new flag is set while the switch is off")

    def test_default_env_value_is_off(self):
        self.assertFalse(S._infeasible_skip_armed())

    def test_on_an_infeasible_head_does_not_block_a_younger_grant(self):
        head, sched = self._head()
        S._INFEASIBLE["weg2-0-9"] = True
        r, ggn = self._grant("weg2-0-10", True, sched)
        self.assertEqual(ggn, 1, "the younger rid reaches group_grant instead of the older-head hold")
        self.assertFalse([l for l in self.lines if "hold=older-head" in l])
        self.assertTrue(S._INFEASIBLE["weg2-0-10"], "the younger one is itself infeasible here (need 4080 > pool 3100)")

    def test_on_a_feasible_older_head_still_holds(self):
        head, sched = self._head()
        S._INFEASIBLE["weg2-0-9"] = False
        r, ggn = self._grant("weg2-0-10", True, sched)
        self.assertEqual((r, ggn), (0, 0))
        self.assertTrue([l for l in self.lines if "hold=older-head" in l])

    def test_on_the_head_keeps_retrying_itself_and_refreshes_its_flag(self):
        head, sched = self._head()
        r, ggn = self._grant("weg2-0-9", True, sched)         # the head's own attempt (no older waiter)
        self.assertEqual(ggn, 1)
        self.assertTrue(S._INFEASIBLE["weg2-0-9"])
        # D yields everything and the pool grows to fit: the flag turns False on the next attempt
        _ledger(self.led[0], 6000, d_committed=0)
        r, ggn = self._grant("weg2-0-9", True, sched)
        self.assertFalse(S._INFEASIBLE.get("weg2-0-9", False))

    def test_a_grant_clears_the_flag(self):
        S._INFEASIBLE["weg2-0-77"] = True
        S._WAITS["weg2-0-77"] = [999.0, 999.0, 1.0, 3, 0]
        S._wait_granted("weg2-0-77")
        self.assertNotIn("weg2-0-77", S._INFEASIBLE)

    def test_marker_is_written_with_the_switch_off_too(self):
        r, ggn = self._grant("weg2-0-11", False, _Sched())
        self.assertEqual(r, 0)
        inf = [l for l in self.lines if "#1640 GRANT-INFEASIBLE" in l]
        self.assertEqual(len(inf), 1, "the first wait writes the marker (backoff curve)")
        self.assertIn("rid=weg2-0-11", inf[0])
        self.assertIn("0:need=4080,pool=3100", inf[0])
        self.assertEqual(S._INFEASIBLE, {}, "switch off: only the lazily computed log line, no state")

    def test_a_merely_short_grant_writes_no_marker(self):
        _ledger(self.led[0], 6000, d_committed=5000)          # free 1000, need 4080 <= 1000 + 5000
        r, ggn = self._grant("weg2-0-12", False, _Sched())
        self.assertEqual(r, 0)
        self.assertEqual([l for l in self.lines if "GRANT-INFEASIBLE" in l], [])

    def test_a_non_pp0_rank_evaluates_and_sets_nothing(self):
        r, ggn = self._grant("weg2-0-13", True, _Sched(pp_rank=1))
        self.assertIsNone(r)
        self.assertEqual((ggn, S._INFEASIBLE), (0, {}))


class OlderWaitsDirect(CustomTestCase):
    def setUp(self):
        S._reset_wait_log()

    def test_skip_flag_is_honoured_only_when_armed(self):
        h1, h2 = _Req("weg2-0-1", wait=True), _Req("weg2-0-2", wait=True)
        sched = _Sched(held={"a": h1, "b": h2})
        S._WAITS["weg2-0-1"] = [10.0, 10.0, 1.0, 1, 0]
        S._WAITS["weg2-0-2"] = [20.0, 20.0, 1.0, 1, 0]
        S._WAITS["weg2-0-3"] = [30.0, 30.0, 1.0, 1, 0]
        S._INFEASIBLE["weg2-0-1"] = True
        with mock.patch.object(S, "_infeasible_skip_armed", lambda: False):
            self.assertEqual([k for k, _t in S._older_waits(sched, "weg2-0-3")], ["weg2-0-1", "weg2-0-2"])
        with mock.patch.object(S, "_infeasible_skip_armed", lambda: True):
            self.assertEqual([k for k, _t in S._older_waits(sched, "weg2-0-3")], ["weg2-0-2"])


if __name__ == "__main__":
    import unittest

    unittest.main()
