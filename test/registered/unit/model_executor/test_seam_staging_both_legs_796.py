"""#796 -- the KV pool must be sized for the worse flip leg, not one of them.

THE DEFECT, observed end-to-end on this rig (boot_restore_797.log).
``_seam_staging_ask_bytes`` charged the sizing budget for the ``pp_to_tp``
staging leg ONLY. ``tp_to_pp`` abstained by design, on the argument that its
pool is about to become active again and ``recover_kv_backing`` would undo any
shrink inside the same flip -- so charging for it would hold memory against a
payment nobody makes.

That argument assumes recovery can always pay. It cannot.
``KvBackingRelief.recover`` is bounded by the CORRIDOR LAW, not by what the
seam needs, so once the pool has been sized to rest near that law floor --
precisely what happens when nothing reserves slack for the un-charged leg --
recovery has nothing left to give and logs "recovery deferred: N MiB free
leaves nothing above the 1024 MiB corridor law to re-commit with". The
un-funded leg can then never arm.

What that cost, measured: eight consecutive ``tp_to_pp`` attempts abandoned at
a stable 127-164 MiB shortfall ("staging 2068 MiB needed but only 1911 MiB is
spendable"), then "SEAM UNFUNDABLE -- PHASE FLIP STOOD DOWN". The instance was
pinned in one layout for the rest of its life and began refusing long-context
requests outright -- while the other two ranks sat on 1826 and 2866 MiB free.

The fix projects BOTH directions and charges the LARGER. ``max`` and not a sum,
because only one direction stages at a time: the pool must cover the worse leg,
not both at once. That is deliberately unlike the arming-floor + staging pair,
which ``pool_flip_posts_bytes`` ADDS precisely because those two are needed at
the same instant.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MIB = 1024 * 1024


class _Runtime:
    """Stands in for PhaseFlipRuntime. Records which directions were asked."""

    def __init__(self, per_direction, raises=False):
        self._per_direction = per_direction
        self._raises = raises
        self.asked = []

    def project_staging_bytes(self, direction: str, n_slots: int) -> int:
        if self._raises:
            raise RuntimeError("projection exploded")
        self.asked.append((direction, n_slots))
        return self._per_direction[direction]


def _runner(per_direction=None, *, slots=8, raises=False, warm=None):
    runtime = (
        _Runtime(per_direction, raises=raises) if per_direction is not None else None
    )
    stub = SimpleNamespace(
        phase_flip_runtime=runtime,
        server_args=SimpleNamespace(max_running_requests=slots),
    )
    stub._seam_reserve = lambda: SimpleNamespace(staging_ask_bytes=warm or 0)
    return stub, runtime


def _ask(stub):
    return ModelRunnerKVCacheMixin._seam_staging_ask_bytes(stub)


class BothLegsArePriced796(CustomTestCase):
    def test_both_directions_are_projected(self):
        stub, rt = _runner({"pp_to_tp": 100 * _MIB, "tp_to_pp": 200 * _MIB})
        _ask(stub)
        self.assertEqual(sorted(d for d, _ in rt.asked), ["pp_to_tp", "tp_to_pp"])

    def test_the_larger_leg_is_charged_when_it_is_tp_to_pp(self):
        """THE DEFECT CASE. This is the leg that used to abstain, and the one
        the rig's flip actually starved on."""
        stub, _ = _runner({"pp_to_tp": 100 * _MIB, "tp_to_pp": 200 * _MIB})
        value, _prov = _ask(stub)
        self.assertEqual(value, 200 * _MIB)

    def test_the_larger_leg_is_charged_when_it_is_pp_to_tp(self):
        """The regression guard: where pp_to_tp already dominated, the charge
        is exactly what it was before #796."""
        stub, _ = _runner({"pp_to_tp": 300 * _MIB, "tp_to_pp": 120 * _MIB})
        value, _prov = _ask(stub)
        self.assertEqual(value, 300 * _MIB)

    def test_the_charge_is_the_max_not_the_sum(self):
        """Only one direction stages at a time. Summing would hold back memory
        for a simultaneity that never occurs."""
        stub, _ = _runner({"pp_to_tp": 300 * _MIB, "tp_to_pp": 200 * _MIB})
        value, _prov = _ask(stub)
        self.assertEqual(value, 300 * _MIB)
        self.assertNotEqual(value, 500 * _MIB)

    def test_equal_legs_charge_that_value_once(self):
        stub, _ = _runner({"pp_to_tp": 150 * _MIB, "tp_to_pp": 150 * _MIB})
        value, _prov = _ask(stub)
        self.assertEqual(value, 150 * _MIB)

    def test_the_slot_count_reaches_both_projections(self):
        stub, rt = _runner({"pp_to_tp": 1, "tp_to_pp": 2}, slots=13)
        _ask(stub)
        self.assertEqual([n for _, n in rt.asked], [13, 13])


class TheProvenanceNamesTheLegAndBothNumbers796(CustomTestCase):
    """An operator has to be able to tell an exact projection from a cold zero,
    and now also WHICH leg set the price, without reading the code."""

    def test_it_names_the_worst_leg(self):
        stub, _ = _runner({"pp_to_tp": 100 * _MIB, "tp_to_pp": 200 * _MIB})
        _value, prov = _ask(stub)
        self.assertIn("tp_to_pp", prov)
        self.assertIn("worst of", prov)

    def test_it_carries_both_measurements(self):
        stub, _ = _runner({"pp_to_tp": 111, "tp_to_pp": 222})
        _value, prov = _ask(stub)
        self.assertIn("111", prov)
        self.assertIn("222", prov)


class TheFallbacksAreUntouched796(CustomTestCase):
    def test_a_failing_projection_never_fails_the_boot(self):
        stub, _ = _runner({"pp_to_tp": 1, "tp_to_pp": 2}, raises=True)
        value, prov = _ask(stub)
        self.assertEqual(value, 0)
        self.assertIn("projection unavailable", prov)

    def test_the_warm_record_is_used_when_there_is_no_runtime(self):
        stub, _ = _runner(None, warm=77 * _MIB)
        value, prov = _ask(stub)
        self.assertEqual(value, 77 * _MIB)
        self.assertIn("warm seam record", prov)

    def test_the_cold_case_is_zero_and_says_so(self):
        stub, _ = _runner(None)
        value, prov = _ask(stub)
        self.assertEqual(value, 0)
        self.assertIn("cold", prov)


class CanFail796(CustomTestCase):
    """Must go red if the charge reverts to a single hardcoded leg."""

    def test_a_pp_to_tp_only_charge_is_detectable(self):
        stub, _ = _runner({"pp_to_tp": 100 * _MIB, "tp_to_pp": 900 * _MIB})
        value, _prov = _ask(stub)
        self.assertNotEqual(
            value,
            100 * _MIB,
            "the charge equals the pp_to_tp leg while tp_to_pp is 9x larger: "
            "the tp_to_pp leg is being ignored again and #796 is unfixed",
        )
        self.assertEqual(value, 900 * _MIB)

    def test_the_tp_to_pp_leg_is_actually_queried(self):
        stub, rt = _runner({"pp_to_tp": 1, "tp_to_pp": 2})
        _ask(stub)
        self.assertIn(
            "tp_to_pp",
            [d for d, _ in rt.asked],
            "tp_to_pp was never projected: the sizing budget cannot know what "
            "that leg costs",
        )


if __name__ == "__main__":
    unittest.main()
