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
"""PP x storage-backed HiCache arms again -- and the record of why, twice.

THIS CONFIGURATION HAS BEEN GATED AND UNGATED THREE TIMES. The history is the
test, because the failure mode is procedural, not mechanical.

1. Originally refused: PP + a storage backend wedged at warmup (#630).
2. #703 lifted it, arguing "the wedge's root fix is 9da9dfd025 (bounded
   collectives)" and "a guard cannot be justified by a defect that a green suite
   says is fixed". Both halves were wrong. 9da9dfd025 BOUNDED the wait; it did
   not repair the desync. And the suite it rested on
   (test_hicache_bounded_waits_630.py) asserts only that bounded calls raise ON
   SCHEDULE against mocked Work objects -- it proves the wait expires, never
   that two ranks exchange anything.
3. Restored 2026-08-17 after the configuration wedged on metal for eleven
   minutes: PP0 pp_sync/isend[0]->pp1, PP1 recv<-pp0, PP2 recv<-pp1, all 649 s,
   every op correctly posted. The restored clause named its own exit condition:
   "Remove this only when a test proves two ranks RENDEZVOUS, not when one
   proves a wait expires."
4. Lifted again -- here -- because that test now exists AND the defect is rooted:
   bounded_wait polled is_completed() and only called wait() after the poll
   succeeded; is_completed() REPORTS while wait() DRIVES, so two polling peers
   never advanced the exchange. The bound was the livelock. See
   test_pp_sync_rendezvous_630.py, which runs THREE REAL PROCESSES over a REAL
   gloo group.

So these tests pin the CURRENT state (no refusal) while
test_pp_sync_rendezvous_630.py pins the property that earns it. If this
configuration wedges again, restore the clause -- and do not accept a green mock
suite as grounds to lift it a third time.
"""

import types
import unittest

from sglang.srt.managers.phase_flip_runtime import flip_blocking_guards
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1)


class _TreeCache:
    def all_values_flatten(self):  # pragma: no cover - existence is the point
        return ()


def _scheduler(*, pp_size=3, hierarchical=True, backend="file"):
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


class TheConfigurationArms(unittest.TestCase):
    def test_pp_with_a_storage_backend_no_longer_refuses(self):
        """The 2026-08-17 boot shape. Green only because the desync is rooted."""
        self.assertEqual([], _hicache_guards(_scheduler()))

    def test_no_backend_and_single_stage_also_arm(self):
        for kw in ({"backend": None}, {"pp_size": 1}, {"hierarchical": False}):
            with self.subTest(**kw):
                self.assertEqual([], _hicache_guards(_scheduler(**kw)))


class TheRendezvousProofExists(unittest.TestCase):
    """The lift is only legitimate while its evidence is present.

    Deleting the rendezvous suite would silently return this configuration to
    the state that wedged -- ungated, with nothing proving ranks exchange.
    """

    def test_the_three_process_rendezvous_suite_is_present(self):
        import pathlib

        suite = (
            pathlib.Path(__file__).resolve().parents[1]
            / "mem_cache"
            / "test_pp_sync_rendezvous_630.py"
        )
        self.assertTrue(suite.is_file(), f"missing the lift's evidence: {suite}")
        # Read rather than import: that module spawns real processes, and this
        # assertion is about the evidence EXISTING, not about re-running it.
        text = suite.read_text()
        for name in ("ThreeRealRanksExchangeValues", "TheBoundStillFires"):
            self.assertIn(name, text, f"{name} is the evidence for lifting the guard")


if __name__ == "__main__":
    unittest.main()
