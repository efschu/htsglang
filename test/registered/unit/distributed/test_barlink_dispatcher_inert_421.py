"""#421 F9: which half of the barlink path dispatcher is inert, and why.

The audit recorded two facts about
``distributed/device_communicators/barlink_path_dispatcher.py`` and left one
of them as an OPEN QUESTION. Both are settled here, in the direction of "this
is the #279 skeleton's intended state", and pinned so the state stays visible
instead of living in a comment nobody re-reads:

* ``PathProfile.transport_hint`` is never written by a production
  construction site, so ``refine_transport_choice`` -- the #240 ``_select``
  hook -- can never act on a measured decision. Pinned as an ABSENCE with a
  named remedy: wiring it is #279's actuation slice, and the pin going red is
  the good outcome.
* ``PathProfile.saturation_threshold`` has no writer either, so it is
  permanently 1.0. The audit did not trace whether that makes the overflow
  re-route unreachable. It does not: ``_utilization_locked`` returns the
  injected sensor's value unclamped, so a sensor reporting exactly 1.0 fires
  the branch. The branch is live code with a threshold that is effectively
  off, which is a defensible default (a threshold below 1 is a measured
  figure and belongs to the same #279 slice as the sensor) -- not dead code.

Hermetic: pure CPU, no torch.distributed, no GPU.
"""

import ast
import pathlib
import unittest

from sglang.srt.distributed.device_communicators.barlink_path_dispatcher import (
    HINT_GLOO,
    HINT_TRANSPORT,
    PROVENANCE_MEASURED,
    DispatchRequest,
    PathDispatcher,
    PathProfile,
    RatePoint,
    refine_transport_choice,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_RATES = "python/sglang/srt/distributed/device_communicators/barlink_path_rates.py"


def _measured(name, base_ms, per_byte_ms, **kw):
    p = PathProfile(
        name=name,
        provenance=PROVENANCE_MEASURED,
        points=[RatePoint(1 << 20, base_ms + per_byte_ms * (1 << 20))],
        base_ms=base_ms,
        per_byte_ms=per_byte_ms,
        **kw,
    )
    return p


class TestSaturationThresholdIsLiveButEffectivelyOff(CustomTestCase):
    """Answers the audit's open question with a falsifier in both
    directions: the overflow tier fires, and the default keeps it quiet."""

    def _two_path_dispatcher(self, **kw):
        d = PathDispatcher()
        d.register_path(_measured("fast", 0.1, 1e-9, **kw))
        d.register_path(_measured("slow", 0.2, 2e-9, **kw))
        return d

    def test_overflow_branch_is_reachable_at_exactly_one(self):
        d = self._two_path_dispatcher()
        d.set_saturation_sensor(lambda name: 1.0 if name == "fast" else 0.0)
        decision = d.decide(DispatchRequest("collective", 1 << 20))
        self.assertEqual(decision.path, "slow")
        self.assertTrue(decision.overflowed)

    def test_below_one_never_overflows_with_the_default_threshold(self):
        d = self._two_path_dispatcher()
        d.set_saturation_sensor(lambda name: 0.99)
        decision = d.decide(DispatchRequest("collective", 1 << 20))
        self.assertEqual(decision.path, "fast")
        self.assertFalse(decision.overflowed)

    def test_a_measured_threshold_below_one_would_work(self):
        """The field is not vestigial: the #279 measured slice can set it and
        the tier starts working, with no other change."""
        d = PathDispatcher()
        d.register_path(_measured("fast", 0.1, 1e-9, saturation_threshold=0.8))
        d.register_path(_measured("slow", 0.2, 2e-9, saturation_threshold=0.8))
        d.set_saturation_sensor(lambda name: 0.9 if name == "fast" else 0.0)
        decision = d.decide(DispatchRequest("collective", 1 << 20))
        self.assertEqual(decision.path, "slow")
        self.assertTrue(decision.overflowed)


class TestTransportHintIsStillUnwired(CustomTestCase):
    """PIN OF AN ABSENCE. When one of these fails somebody wired the #279
    actuation hook -- delete the pin and the F9 row of the audit, do NOT
    widen it."""

    def test_no_production_construction_site_passes_a_hint(self):
        tree = ast.parse((_REPO_ROOT / _RATES).read_text())
        sites = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "PathProfile"
            ):
                if any(kw.arg == "transport_hint" for kw in node.keywords):
                    sites.append(node.lineno)
        self.assertEqual(
            sites,
            [],
            "GOOD NEWS: the barlink path dispatcher now gets an actuation "
            f"hint from a production rate source (lines {sites}). #421 "
            "finding F9 is fixed -- delete this pin.",
        )

    def test_a_hintless_measured_decision_keeps_the_status_quo(self):
        """The consequence of the absence, demonstrated rather than asserted:
        a fully measured, non-status-quo decision still changes nothing."""
        d = PathDispatcher()
        d.register_path(_measured("fast", 0.1, 1e-9))
        d.register_path(_measured("slow", 0.2, 2e-9))
        decision = d.decide(DispatchRequest("collective", 1 << 20))
        self.assertFalse(decision.status_quo)
        self.assertIsNone(d.transport_hint(decision.path))
        sentinel = object()
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.barlink_path_dispatcher",
            level="WARNING",
        ):
            self.assertIs(
                refine_transport_choice(d, "all_reduce", 1 << 20, sentinel),
                sentinel,
            )

    def test_the_hook_does_act_once_a_hint_exists(self):
        """The other half of the falsifier: the hook is not broken, it is
        merely never handed a hint. Both hint values are exercised."""
        gloo = PathDispatcher()
        gloo.register_path(_measured("bar1", 0.1, 1e-9, transport_hint=HINT_GLOO))
        gloo.register_path(_measured("nccl", 0.2, 2e-9))
        self.assertIsNone(
            refine_transport_choice(gloo, "all_reduce", 1 << 20, object())
        )

        keep = PathDispatcher()
        keep.register_path(_measured("bar1", 0.1, 1e-9, transport_hint=HINT_TRANSPORT))
        keep.register_path(_measured("nccl", 0.2, 2e-9))
        sentinel = object()
        self.assertIs(
            refine_transport_choice(keep, "all_reduce", 1 << 20, sentinel),
            sentinel,
        )


if __name__ == "__main__":
    unittest.main()
