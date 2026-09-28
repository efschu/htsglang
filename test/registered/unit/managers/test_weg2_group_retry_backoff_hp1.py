"""HP1 retry cap (rc12z20, boot dkrnfh91dprsavisnoadoptstbar1dauer09281220 on
3a86888ba5, D 12:30:32-12:31:36, rid weg2-8-2): a request DEFERRED because the
group MIN cut its store read (host_pool_shortfall / store_prefix_short) was
re-issued -- intake re-match plus the #580 vote collective -- on EVERY
scheduler pass: 638 group truncations in 64 s, 22 M tokens attempted, the
decode round's host share ~13 -> ~32 ms (DECODE-HOST-SPLIT sched_ms 3 -> 24)
while D decoded at bs1.

RED on ebeccfd190: no ``_hp1_group_retry_due``. GREEN: a group-arm mark
re-issues on a pass back-off (1, 2, 4, 8, 16, then every 16th), counted in
retry passes of the mark (rank-identical), and every skipped pass still runs
the arm's re-defer step so the #1317k standstill bound keeps counting passes.
"""

from __future__ import annotations

import types
import unittest


def _due(sched, req):
    from sglang.srt.managers.scheduler import _hp1_group_retry_due

    return _hp1_group_retry_due(sched, req)


class _Sched:
    def __init__(self):
        self.arm_calls = []

    def _apply_group_shortfall_deferral(self, req, rid, span, marked, site, reason=None):
        self.arm_calls.append((rid, marked, site, reason))
        return "deferred"


class TestRetryBackoff(unittest.TestCase):
    def test_rc12z20_group_mark_retries_on_a_pass_backoff(self):
        s = _Sched()
        req = types.SimpleNamespace(rid="weg2-8-2", prefetch_deferred="host_pool_shortfall",
                                    prefetch_defer_since=1.0, _prefetch_span_tokens=33600)
        due = [n for n in range(1, 65) if _due(s, req)]
        self.assertEqual(due, [1, 2, 4, 8, 16, 32, 48, 64],
                         "rc12z20: 64 passes = 64 re-votes; the cap leaves 8")

    def test_a_skipped_pass_runs_the_arms_redefer_step(self):
        s = _Sched()
        req = types.SimpleNamespace(rid="weg2-8-2", prefetch_deferred="host_pool_shortfall",
                                    prefetch_defer_since=1.0, _prefetch_span_tokens=33600)
        n_due = sum(_due(s, req) for _ in range(64))
        self.assertEqual(len(s.arm_calls), 64 - n_due, "every skipped pass is observed")
        self.assertEqual(set(c[1:] for c in s.arm_calls),
                         {(True, "retry_backoff", "host_pool_shortfall")})

    def test_fresh_mark_restarts_and_rate_arm_is_untouched(self):
        s = _Sched()
        req = types.SimpleNamespace(rid="r", prefetch_deferred="store_prefix_short",
                                    prefetch_defer_since=1.0)
        [_due(s, req) for _ in range(5)]
        req.prefetch_defer_since = 2.0
        self.assertTrue(_due(s, req), "a fresh mark re-issues at once")
        req.prefetch_deferred = "rate_limited"
        self.assertTrue(all(_due(s, req) for _ in range(10)))

    def test_backoff_is_a_pure_function_of_the_pass_count(self):
        # Rank-uniformity: three ranks with the same mark skip the same passes
        # even though each stamped its own clock into prefetch_defer_since.
        seqs = []
        for rank in range(3):
            s = _Sched()
            req = types.SimpleNamespace(rid="r", prefetch_deferred="host_pool_shortfall",
                                        prefetch_defer_since=100.0 + rank)
            seqs.append([_due(s, req) for _ in range(100)])
        self.assertEqual(seqs[0], seqs[1])
        self.assertEqual(seqs[0], seqs[2])

    def test_the_retry_loop_consults_the_cap(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._retry_deferred_prefetches)
        i = src.index("_hp1_group_retry_due(self, req)")
        self.assertLess(i, src.index("self._prefetch_kvcache(req)"))


if __name__ == "__main__":
    unittest.main()
