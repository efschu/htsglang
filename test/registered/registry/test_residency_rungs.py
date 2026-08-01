# SPDX-License-Identifier: Apache-2.0
"""#305 cut 1: the residency ladder view over the engine registry.

The registry (#333-M1) already had four residency states, a ledger, pin as a
spec property and a ``/v1/models`` listing. This cut adds the three things
``DESIGN_305_multi_model_serving.md`` asks for and that registry did not
answer, and each gets its own tests here:

1. the FIFTH state -- "registered, nothing staged" separated from "cold but
   previously staged";
2. a promote-cost CLASS a-priori, kept in a different field from the
   registry's measured figure so a class can never be read as a measurement;
3. cross-geometry labelling at REGISTRATION time, plus the pin contract --
   an unhonourable pin fails AT PIN TIME, which the shipped `post_pin` did
   not do (it set the flag unconditionally).

Hermetic: pure functions over the state enum, no server, no card.
"""

import unittest

from sglang.srt.registry.ledger import TenantState
from sglang.srt.registry.rungs import (
    LADDER,
    Rung,
    cross_geometry_label,
    pin_refusal_reason,
    promote_cost_of,
    rung_extension,
    rung_of,
    transition_refusal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRungMapping(CustomTestCase):
    def test_the_four_registry_states_map_onto_the_ladder(self):
        self.assertEqual(rung_of(TenantState.HOT), Rung.HOT)
        self.assertEqual(rung_of(TenantState.WARM_GPU), Rung.TEIL_HOT)
        self.assertEqual(rung_of(TenantState.WARM_HOST), Rung.WARM)
        self.assertEqual(rung_of(TenantState.COLD), Rung.COLD)

    def test_warm_gpu_is_the_teil_hot_rung(self):
        # The design's TEIL-HOT is "weights resident, pools reduced" -- the
        # state that still holds device memory without serving.
        ext = rung_extension(TenantState.WARM_GPU)
        self.assertEqual(ext["rung"], Rung.TEIL_HOT)
        self.assertTrue(ext["gpu_resident"])

    def test_warm_host_holds_no_device_memory(self):
        self.assertFalse(rung_extension(TenantState.WARM_HOST)["gpu_resident"])

    def test_plain_strings_are_accepted(self):
        self.assertEqual(rung_of("HOT"), Rung.HOT)

    def test_an_unknown_state_is_an_error_not_a_guess(self):
        with self.assertRaises(ValueError):
            rung_of("LUKEWARM")


class TestTheFifthState(CustomTestCase):
    """"Registered" and "cold" answer different client questions."""

    def test_never_staged_cold_is_REGISTERED(self):
        self.assertEqual(
            rung_of(TenantState.COLD, ever_staged=False, reserved_bytes=0),
            Rung.REGISTERED,
        )

    def test_previously_staged_cold_stays_COLD(self):
        # It has an image on disk to resume from; the never-staged one has
        # only a config. Same state enum, different promise to a client.
        self.assertEqual(
            rung_of(TenantState.COLD, ever_staged=True), Rung.COLD
        )

    def test_reserved_bytes_imply_staging_even_if_the_flag_says_otherwise(self):
        # Defensive: bytes on a card are evidence, a flag is bookkeeping.
        self.assertEqual(
            rung_of(TenantState.COLD, ever_staged=False, reserved_bytes=1 << 20),
            Rung.COLD,
        )

    def test_only_cold_can_become_registered(self):
        for st in (TenantState.HOT, TenantState.WARM_GPU, TenantState.WARM_HOST):
            self.assertNotEqual(rung_of(st, ever_staged=False), Rung.REGISTERED)


class TestPromoteCostClass(CustomTestCase):
    def test_every_rung_has_a_cost_with_a_basis(self):
        for rung in (Rung.HOT, Rung.TEIL_HOT, Rung.WARM, Rung.COLD,
                     Rung.REGISTERED):
            cost = promote_cost_of(rung)
            self.assertTrue(cost.basis, f"{rung} has no stated basis")
            self.assertTrue(cost.seconds)

    def test_the_costs_are_the_ladders_measured_record(self):
        self.assertEqual(promote_cost_of(Rung.TEIL_HOT).seconds, "<1")
        self.assertEqual(promote_cost_of(Rung.WARM).seconds, "3-6")
        self.assertEqual(promote_cost_of(Rung.COLD).seconds, "12-20")
        self.assertIn("#297", promote_cost_of(Rung.TEIL_HOT).basis)
        self.assertIn("#89", promote_cost_of(Rung.COLD).basis)

    def test_a_class_is_never_marked_measured(self):
        # The whole point of the separate field: an a-priori class must not be
        # readable as a per-engine measurement.
        for cost in LADDER.values():
            self.assertFalse(cost.measured)

    def test_the_extension_marks_the_class_as_unmeasured(self):
        ext = rung_extension(TenantState.COLD)
        self.assertFalse(ext["promote_cost_class"]["measured"])
        self.assertIn("basis", ext["promote_cost_class"])

    def test_an_unknown_rung_has_no_cost(self):
        with self.assertRaises(ValueError):
            promote_cost_of("LUKEWARM")


class TestCrossGeometryLabelling(CustomTestCase):
    def test_matching_geometry_is_not_labelled(self):
        self.assertIsNone(cross_geometry_label("tp3-uneven", "tp3-uneven"))

    def test_differing_geometry_is_labelled_with_the_floor_and_instrument(self):
        label = cross_geometry_label("tp2", "tp3-uneven")
        self.assertTrue(label["cross_geometry"])
        self.assertEqual(label["floor_seconds"], "12-20")
        self.assertIn("#329", label["instrument"])
        self.assertIn("#309", label["reason"])  # names what does NOT apply

    def test_unknown_geometry_is_not_declared_cross(self):
        # Guessing would attach a 12-20 s warning to every engine whose label
        # an operator simply did not fill in.
        self.assertIsNone(cross_geometry_label(None, "tp3-uneven"))
        self.assertIsNone(cross_geometry_label("tp2", None))

    def test_the_label_rides_in_the_extension(self):
        ext = rung_extension(
            TenantState.COLD, engine_geometry="tp2", active_geometry="tp3"
        )
        self.assertIn("geometry", ext)
        ext2 = rung_extension(
            TenantState.COLD, engine_geometry="tp3", active_geometry="tp3"
        )
        self.assertNotIn("geometry", ext2)


class TestPinContract(CustomTestCase):
    """Pin blocks demotion; it cannot force promotion past the ledger."""

    def test_unpin_is_always_honoured(self):
        self.assertIsNone(
            pin_refusal_reason(rung=Rung.COLD, pinned=False, can_fund=False)
        )

    def test_pinning_a_resident_engine_is_honoured(self):
        for rung in (Rung.HOT, Rung.TEIL_HOT):
            self.assertIsNone(
                pin_refusal_reason(rung=rung, pinned=True, can_fund=False)
            )

    def test_an_unfundable_pin_is_refused_AT_PIN_TIME(self):
        # The shipped post_pin set the flag unconditionally; this is the gap.
        reason = pin_refusal_reason(
            rung=Rung.COLD, pinned=True, can_fund=False,
            ledger_detail="needs 18000 MiB, 4200 free",
        )
        self.assertIsNotNone(reason)
        self.assertIn("cannot fund", reason)
        self.assertIn("at pin time", reason)

    def test_the_refusal_carries_the_ledger_detail(self):
        reason = pin_refusal_reason(
            rung=Rung.WARM, pinned=True, can_fund=False,
            ledger_detail="needs 18000 MiB, 4200 free",
        )
        self.assertIn("18000 MiB", reason)

    def test_a_fundable_cold_pin_is_honoured(self):
        self.assertIsNone(
            pin_refusal_reason(rung=Rung.COLD, pinned=True, can_fund=True)
        )

    def test_a_cross_geometry_pin_is_refused_with_its_own_reason(self):
        reason = pin_refusal_reason(
            rung=Rung.COLD, pinned=True, can_fund=True, cross_geometry=True
        )
        self.assertIsNotNone(reason)
        self.assertIn("#329", reason)
        self.assertNotIn("cannot fund", reason)  # a different refusal


class TestNoMovementAdvertised(CustomTestCase):
    """#111: the seam exists, the movement does not, and the error says so."""

    def test_every_target_names_the_cut_that_would_implement_it(self):
        self.assertIn("cut 2", transition_refusal(Rung.HOT))
        self.assertIn("cut 2", transition_refusal(Rung.TEIL_HOT))
        self.assertIn("cut 3", transition_refusal(Rung.WARM))
        self.assertIn("cut 5", transition_refusal(Rung.COLD))

    def test_the_refusal_names_the_instrument_for_cut_2(self):
        msg = transition_refusal(Rung.TEIL_HOT)
        self.assertIn("#330", msg)
        self.assertIn("#309", msg)


class TestExtensionShape(CustomTestCase):
    def test_the_block_is_json_shaped_and_complete(self):
        import json

        ext = rung_extension(
            TenantState.WARM_HOST, pinned=True,
            engine_geometry="tp2", active_geometry="tp3",
        )
        payload = json.loads(json.dumps(ext))
        for key in ("rung", "gpu_resident", "pinned", "promote_cost_class",
                    "geometry"):
            self.assertIn(key, payload)
        self.assertTrue(payload["pinned"])
        self.assertEqual(payload["rung"], Rung.WARM)


if __name__ == "__main__":
    unittest.main()
