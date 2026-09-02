"""#1068 slice 3 (T13): the post-cutover readmit re-issues the WHOLE
run-willing population and prints ONE aggregate line (L7, G11).

RED on 846c6797b9: `_post_cutover_readmit` calls
`readmit_seam_residents(list(released))` without `requeue_waiting`, reads no
`last_seam_readmit`, and its #1066 line carries neither `issued=` nor
`declined=` nor `reasons=`.

The runtime is driven through the REAL `PhaseFlipRuntime._post_cutover_readmit`
bound to a bare instance (`__new__`), with a scheduler stand-in that records
the call and publishes the summary the real `readmit_seam_residents` publishes.
"""

import types
import unittest

from sglang.srt.managers import phase_flip_runtime as pfr
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Sched:
    def __init__(self, summary, readmitted):
        self.calls = []
        self._summary = summary
        self._readmitted = readmitted
        self.tree_cache = types.SimpleNamespace(
            cache_controller=types.SimpleNamespace(
                mem_pool_host=types.SimpleNamespace(size=366211)
            )
        )

    def readmit_seam_residents(self, reqs, requeue_waiting=False):
        self.calls.append((list(reqs), requeue_waiting))
        self.last_seam_readmit = dict(self._summary)
        return self._readmitted


def _runtime(sched, stash):
    rt = pfr.PhaseFlipRuntime.__new__(pfr.PhaseFlipRuntime)
    rt._census_scheduler = sched
    rt._pending_seam_readmit = stash
    return rt


class TestAggregateLine(CustomTestCase):
    def _drive(self, summary, released, readmitted):
        sched = _Sched(summary, readmitted)
        rt = _runtime(sched, (list(released), len(released)))
        with self.assertLogs(pfr.logger, level="INFO") as caught:
            pfr.logger.info("probe: post-cutover driven")
            rt._post_cutover_readmit("pp_to_tp")
        return sched, [ln for ln in caught.output if "POST-CUTOVER FRESH-FETCH" in ln]

    def test_aggregate_line_names_population_and_verdicts(self):
        # T13: L7 with issued= / declined= / reasons= / dropped_by_queue_limit=.
        summary = {
            "retracted": 3,
            "requeued": 2,
            "residents": 3,
            "occupants": 3,
            "verdicts": {"issued": 4, "declined:store_absent": 1},
            "queue_before": 3,
            "queue_after": 5,
            "dropped_by_queue_limit": 1,
        }
        released = [types.SimpleNamespace(rid=f"r{i}") for i in range(3)]
        sched, lines = self._drive(summary, released, readmitted=3)
        self.assertEqual(len(lines), 1, lines)
        line = lines[0]
        self.assertIn("re-admitted 3/3 resident(s)", line)
        self.assertIn("re-issued 2/3 queue occupant(s)", line)
        self.assertIn("generation=", line)
        self.assertIn("pool_id=", line)
        self.assertIn("pool_rows=366211", line)
        self.assertIn("issued=4", line)
        self.assertIn("declined=1", line)
        self.assertIn("reasons={'declined:store_absent': 1}", line)
        self.assertIn("dropped_by_queue_limit=1", line)
        # the readmit was asked for the occupants too
        self.assertEqual(len(sched.calls), 1)
        self.assertEqual([r.rid for r in sched.calls[0][0]], ["r0", "r1", "r2"])
        self.assertTrue(sched.calls[0][1], "requeue_waiting=True is the contract")

    def test_no_residents_still_reissues_the_queue(self):
        # A cutover that retracted nothing still dropped every occupant's
        # prefetch record at _reset_full: the readmit must run for them.
        summary = {
            "retracted": 0,
            "requeued": 2,
            "residents": 0,
            "occupants": 2,
            "verdicts": {"issued": 2},
            "queue_before": 2,
            "queue_after": 2,
            "dropped_by_queue_limit": 0,
        }
        sched, lines = self._drive(summary, [], readmitted=0)
        self.assertEqual(len(sched.calls), 1, "the readmit must run with an empty resident list")
        self.assertTrue(sched.calls[0][1])
        self.assertEqual(len(lines), 1)
        self.assertIn("re-admitted 0/0 resident(s) + re-issued 2/2 queue occupant(s)", lines[0])
        self.assertIn("issued=2 declined=0 reasons={}", lines[0])

    def test_the_mismatch_check_stays_on_residents(self):
        summary = {
            "retracted": 1,
            "requeued": 0,
            "residents": 2,
            "occupants": 0,
            "verdicts": {"issued": 1},
            "queue_before": 0,
            "queue_after": 1,
            "dropped_by_queue_limit": 0,
        }
        released = [types.SimpleNamespace(rid="a"), types.SimpleNamespace(rid="b")]
        sched = _Sched(summary, readmitted=1)
        rt = _runtime(sched, (released, 2))
        with self.assertLogs(pfr.logger, level="ERROR") as caught:
            rt._post_cutover_readmit("pp_to_tp")
        self.assertTrue(any("RE-ADMISSION MISMATCH" in ln for ln in caught.output), caught.output)


if __name__ == "__main__":
    unittest.main()
