# Copyright 2023-2024 SGLang Team
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
"""#926: the at-arm census is sampled, and can still fire.

THE INSTRUMENT BECAME THE DEFECT. `_pool_census("at-arm", ...)` walks the pool
and the KV row-ownership map on EVERY arm (#631 J). The 0827 window measured
69 cutovers in five minutes and one of four boots died in a CPU spin whose hot
frame was this census.

The gate is a cadence, not a deletion, so this file pins BOTH halves: it
throttles, and the census remains reachable. An instrument that can no longer
fire would have traded a spin for a blind spot, which is the trade #829 exists
to refuse.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import types
import unittest

from sglang.srt.environ import envs
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.test.test_utils import CustomTestCase


def _holder():
    h = types.SimpleNamespace()
    h._at_arm_census_due = types.MethodType(PhaseFlipRuntime._at_arm_census_due, h)
    return h


class TestTheGateThrottles(CustomTestCase):
    def test_a_fast_arming_instance_is_sampled_not_censused_every_arm(self):
        """The spin, as arithmetic: 69 arms must not be 69 pool walks."""
        h = _holder()
        with envs.SGLANG_PP_ARM_CENSUS_EVERY_N.override(16), (
            envs.SGLANG_PP_ARM_CENSUS_MIN_INTERVAL_S.override(0)
        ):
            fired = sum(1 for _ in range(69) if h._at_arm_census_due())
        self.assertEqual(fired, 4, "expected every-16th sampling over 69 arms")

    def test_the_nth_arm_is_the_one_that_fires(self):
        h = _holder()
        with envs.SGLANG_PP_ARM_CENSUS_EVERY_N.override(4), (
            envs.SGLANG_PP_ARM_CENSUS_MIN_INTERVAL_S.override(0)
        ):
            fired = [i for i in range(1, 13) if h._at_arm_census_due()]
        self.assertEqual(fired, [4, 8, 12])


class TestTheInstrumentCanStillFire(CustomTestCase):
    """The half that keeps this from being a deletion."""

    def test_the_time_budget_alone_admits_the_first_arm(self):
        """A slow-arming instance censuses every arm exactly as before, because
        each one is the first past the interval."""
        h = _holder()
        with envs.SGLANG_PP_ARM_CENSUS_EVERY_N.override(0), (
            envs.SGLANG_PP_ARM_CENSUS_MIN_INTERVAL_S.override(0.0001)
        ):
            import time

            fired = 0
            for _ in range(5):
                if h._at_arm_census_due():
                    fired += 1
                time.sleep(0.001)
        self.assertEqual(fired, 5, "a slow arm cadence must not be throttled")

    def test_both_admissions_off_restores_the_unconditional_census(self):
        """The operator escape hatch: chasing the #631 J page loss needs the
        continuous record back."""
        h = _holder()
        with envs.SGLANG_PP_ARM_CENSUS_EVERY_N.override(0), (
            envs.SGLANG_PP_ARM_CENSUS_MIN_INTERVAL_S.override(0)
        ):
            fired = sum(1 for _ in range(20) if h._at_arm_census_due())
        self.assertEqual(fired, 20)

    def test_a_skipped_arm_is_counted_not_forgotten(self):
        """NEVER SILENTLY FALSE. The skip count must survive to the next
        census, or a sampled window reads exactly like a clean one."""
        h = _holder()
        with envs.SGLANG_PP_ARM_CENSUS_EVERY_N.override(4), (
            envs.SGLANG_PP_ARM_CENSUS_MIN_INTERVAL_S.override(0)
        ):
            for _ in range(3):
                self.assertFalse(h._at_arm_census_due())
            self.assertEqual(getattr(h, "_at_arm_census_skipped", 0), 3)
            self.assertTrue(h._at_arm_census_due())
            self.assertEqual(
                getattr(h, "_at_arm_census_skipped", 0),
                0,
                "the skip count must reset onto the census that reports it",
            )


class TestTheSeamIsWired(CustomTestCase):
    def test_arm_consults_the_gate(self):
        import inspect

        src = inspect.getsource(PhaseFlipRuntime.arm)
        self.assertIn("_at_arm_census_due()", src)
        at = src.index("_at_arm_census_due()")
        self.assertIn('_pool_census("at-arm"', src[at:])


if __name__ == "__main__":
    unittest.main()
