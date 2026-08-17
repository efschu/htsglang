# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#553 planned-cut-3: a relief rung that is chosen must actually be applied.

WHAT WAS ALREADY THERE. #287's ladder ranks rungs and enforces the ordering
invariant; `RELIEF_FEATURES` names the five features a relief rung may
reference; `planner/kv_ladder_table.py` VALIDATES that a rung's
`relief_feature` is one of them; and each feature has a real actuator behind a
flag (`server_args` describes `--max-running-requests-ceiling` as actuating the
floating admission limiter and `--enable-kv-session-offload` as actuating the
#236 spill manager).

WHAT WAS MISSING is the wire between them. Grepped `RELIEF_FEATURES` across the
tree: the only consumers are the ladder that defines it and the table that
checks membership. **Nothing maps a chosen relief rung to the actuator of the
feature it names.** The ladder could say "ascend to the dcp_ratio rung" and
nothing would change.

That is the counter-without-actuator shape, and it is the cut #553's own plan
scoped as Cut 3: "make #287's RELIEF rungs execute, and only those ... an
executor that delegates -- lower the admission cap, arm the spill, shift the DCP
ratio -- needs no new mechanism and no capture change, because relief rungs are
declared 'No KV layout change, hence handover none'."

NOTE ON WHAT SHIPPED UNDER THAT NAME. The commit titled "[#553] Cut 3" delivered
the tenant hot/cold EVENT actuation (`coresidency_policy`), which the plan did
not carry as a numbered cut. It composes the #330 dial and `GdnSlotRuntime`; it
touches no rung. So the planned Cut 3 was never delivered, and this is it.

DELEGATION ONLY, and the refusals are the design. This executor reimplements no
feature: it takes one actuator per feature and reports what the actuator said,
never what the plan intended (#694). An unknown feature refuses by name rather
than silently doing nothing, and a feature with no actuator wired refuses too --
"nothing happened" must never be indistinguishable from "it worked".
"""

from __future__ import annotations

import unittest

from sglang.srt.managers.relief_rung_executor import (
    ReliefActuatorMissing,
    UnknownReliefFeature,
    apply_relief_rung,
)
from sglang.srt.model_executor.kv_pressure_ladder import RELIEF_FEATURES
from sglang.test.test_utils import CustomTestCase


class _Rung:
    """The shape the ladder hands over: a step naming a relief feature."""

    def __init__(self, feature, kind="relief"):
        self.relief_feature = feature
        self.kind = kind


class TestItDelegates(CustomTestCase):
    def test_it_calls_the_actuator_for_the_named_feature(self):
        seen = {}

        def actuator(name):
            def _fn():
                seen["called"] = name
                return True, f"{name} applied"

            return _fn

        actuators = {name: actuator(name) for name in RELIEF_FEATURES}
        result = apply_relief_rung(_Rung("dcp_ratio"), actuators)
        self.assertTrue(result.ok)
        self.assertEqual(seen["called"], "dcp_ratio")
        self.assertEqual(result.feature, "dcp_ratio")

    def test_every_declared_feature_can_be_applied(self):
        """No feature in the vocabulary may be unreachable by construction."""
        actuators = {name: (lambda: (True, "ok")) for name in RELIEF_FEATURES}
        for name in RELIEF_FEATURES:
            with self.subTest(feature=name):
                self.assertTrue(apply_relief_rung(_Rung(name), actuators).ok)

    def test_it_reports_what_the_actuator_said_not_the_plan(self):
        # #694: a total may only come from an actuator report.
        actuators = {"kv_spill": lambda: (False, "spill manager is disabled")}
        result = apply_relief_rung(_Rung("kv_spill"), actuators)
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.detail)


class TestItRefusesRatherThanNoOps(CustomTestCase):
    def test_an_unknown_feature_refuses_by_name(self):
        with self.assertRaises(UnknownReliefFeature) as caught:
            apply_relief_rung(_Rung("teleport_the_kv"), {})
        message = str(caught.exception)
        self.assertIn("teleport_the_kv", message)
        # The refusal must list what IS available, from the ladder's vocabulary.
        self.assertIn("dcp_ratio", message)

    def test_a_feature_with_no_actuator_refuses(self):
        """'Nothing happened' must not be indistinguishable from 'it worked'."""
        with self.assertRaises(ReliefActuatorMissing) as caught:
            apply_relief_rung(_Rung("kv_spill"), {"dcp_ratio": lambda: (True, "")})
        self.assertIn("kv_spill", str(caught.exception))

    def test_a_non_relief_rung_is_refused(self):
        # geometry_flip and external stay stubs; this executor must not be the
        # place someone quietly starts a geometry change.
        with self.assertRaises(UnknownReliefFeature):
            apply_relief_rung(_Rung("dcp_ratio", kind="geometry_flip"), {})


class TestTheVocabularyIsTheLadders(CustomTestCase):
    """The executor must not grow a sixth feature of its own."""

    def test_it_accepts_exactly_the_ladder_vocabulary(self):
        from sglang.srt.managers.relief_rung_executor import supported_features

        self.assertEqual(sorted(supported_features()), sorted(RELIEF_FEATURES))


if __name__ == "__main__":
    unittest.main()
