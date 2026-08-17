# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PP x storage-backed HiCache must refuse to arm: #630 is bounded, not fixed.

THE SPECIMEN, 2026-08-17 10:57:50 -> 11:08:39. A PP=3 line booted with
``--enable-hierarchical-cache --hicache-storage-backend file`` reached readiness
and then never served: health stuck at 503 with nothing logged for eleven
minutes, ending in three simultaneous timeouts on the first event round --

    PP0  pp_sync/isend[0]->pp1   waited 649.1 s
    PP1  pp_sync/recv<-pp0       waited 649.2 s
    PP2  pp_sync/recv<-pp1       waited 649.1 s

All three ranks were INSIDE the collective. PP0's send and PP1's matching
receive were both posted -- same group, same tag, same 0-dim int payload -- and
they did not rendezvous.

WHY THE GUARD CAME BACK, and it was my own removal to withdraw. #703 retired
this clause arguing that "the wedge's root fix is 9da9dfd025 (bounded
collectives)" and that "a guard cannot be justified by a defect that a green
suite says is fixed". Both halves were wrong:

* 9da9dfd025 BOUNDED the wait. It turned a silent two-hour gloo hang into a
  named 600 s error -- which is what fired above. A deadline is not a repair.
* test_hicache_bounded_waits_630.py asserts only that the bounded calls raise
  ON SCHEDULE and post the same operation ``recv`` would, against mocked Work
  objects and fake groups. It proves the wait is bounded. It never proves two
  real ranks rendezvous, which is the thing that fails.

That conflation -- a green suite that proves something weaker than the claim
resting on it -- is what these tests exist to stop repeating.

NARROWER THAN THE ORIGINAL. The pre-#703 clause refused ANY storage backend
under a flip. This one refuses only the combination measured to wedge: a
pipeline (pp_size > 1) carrying a storage-backed tier. The configurations #703
was actually for -- device+host-local tiers, and single-stage flips -- stay
reachable, and the tests below pin that so the guard cannot quietly widen back.
"""

import types
import unittest

from sglang.srt.managers.phase_flip_runtime import flip_blocking_guards
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1)


class _TreeCache:
    """A tree cache that satisfies the unrelated all_values_flatten guard.

    Present so the stand-in does not trip a DIFFERENT clause and make these
    tests pass or fail for a reason that has nothing to do with #630 -- the
    stand-in-AttributeError trap this repo has paid for repeatedly.
    """

    def all_values_flatten(self):  # pragma: no cover - existence is the point
        return ()


def _scheduler(*, pp_size=3, hierarchical=True, backend="file"):
    """Only what flip_blocking_guards reads."""
    server_args = types.SimpleNamespace(
        pp_size=pp_size,
        enable_hierarchical_cache=hierarchical,
        hicache_storage_backend=backend,
    )
    return types.SimpleNamespace(
        server_args=server_args,
        disaggregation_mode=None,
        kv_session_offload=None,
        phase_flip_active_stack=None,
        is_dual_group_lane=False,
        tree_cache=_TreeCache(),
    )


def _hicache_guards(sched):
    return [g for g in flip_blocking_guards(sched) if "hierarchical" in g]


class TheBootedConfigurationIsRefused(unittest.TestCase):
    def test_the_specimen_config_refuses_to_arm(self):
        """PP=3 + file backend: exactly what wedged on metal."""
        guards = _hicache_guards(_scheduler())
        self.assertTrue(guards, "PP x storage-backed HiCache must refuse")
        msg = guards[0]
        self.assertIn("pp_size=3", msg)
        self.assertIn("file", msg)

    def test_the_refusal_says_bounded_not_fixed(self):
        """The message must not repeat the claim that retired the guard.

        Whoever reads this refusal next needs to know that the suite covering
        #630 proves boundedness, not a rendezvous -- otherwise they remove it
        again on the same reasoning.
        """
        msg = _hicache_guards(_scheduler())[0]
        self.assertIn("bounded but NOT fixed", msg)
        self.assertIn("2026-08-17", msg)

    def test_any_storage_backend_counts_not_just_file(self):
        for backend in ("file", "mooncake", "hf3fs", "nixl"):
            with self.subTest(backend=backend):
                self.assertTrue(_hicache_guards(_scheduler(backend=backend)))


class WhatSevenOhThreeUnblockedStaysReachable(unittest.TestCase):
    """CAN-FAIL: a guard that refuses too much is the defect #703 fixed.

    If these go red, the restored clause has widened back into the one that
    made the flip and any prefix cache mutually exclusive -- the state whose
    only answer was running with no cache tier at all.
    """

    def test_device_and_host_local_tier_still_arms(self):
        """Hierarchical cache with NO storage backend is untouched."""
        self.assertEqual([], _hicache_guards(_scheduler(backend=None)))

    def test_a_single_stage_flip_with_a_disk_tier_still_arms(self):
        """The wedge is a PIPELINE defect; pp_size=1 has no pp_sync at all."""
        self.assertEqual([], _hicache_guards(_scheduler(pp_size=1)))

    def test_hierarchical_cache_off_is_untouched(self):
        self.assertEqual([], _hicache_guards(_scheduler(hierarchical=False)))

    def test_pp_two_with_a_disk_tier_is_refused(self):
        """The boundary is pp_size > 1, not pp_size == 3."""
        self.assertTrue(_hicache_guards(_scheduler(pp_size=2)))


if __name__ == "__main__":
    unittest.main()
