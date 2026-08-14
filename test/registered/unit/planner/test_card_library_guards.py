# SPDX-License-Identifier: Apache-2.0
"""The two by-catch defects #584's window reported and did not fix.

DEFECT 1 -- CAPACITY COLLISION. The catalogue keys by card NAME, and the RTX
3080 shipped in a 10 GB and a 20 GB variant. The driver calls both
``NVIDIA GeForce RTX 3080``; ``_canonical`` strips only the vendor words, so
this rig's 20 GB cards land on the 10240 MiB seed entry. #584 corrected the
capacity on the WRITE path (``project_onto_library``) and left the READ path
alone, so every consumer that resolves a live card name against the seed
library -- the roofline's nameplate peaks, the TDP the energy model prices
from, ``compose_rig``'s capacity -- still reads 10240 MiB for a 20480 MiB
card. A wrong capacity does not announce itself: it feeds feasibility and
packing and comes back as a plan that does not fit.

DEFECT 2 -- STALE RATES. #584 measured the three cards and found the borrowed
s50 rates reproduced bandwidth to the decimal and GEMM 12-22 % high, in the
direction and magnitude the 2026-08-05 power-target cut (3080s 320 -> 200 W,
5090 525 -> 400 W) predicts. A GEMM rate is a function of the power limit it
was measured under, and nothing in the artifact recorded that limit -- so
nothing could detect that a persisted rate predates a power-target change.

What these tests pin, in the order the defects bite:

  T1-T5   resolution refuses or selects; it never silently substitutes.
  T6-T7   measured entries win over seeds, and a measured entry that
          disagrees on capacity is still refused.
  T8-T10  the fingerprint round-trips, and an absent one is loud rather than
          absent.
  T11-T13 consumers detect: the roofline declines to price the wrong card,
          and ``--pp-solve-cut`` refuses a rate measured under another power
          limit.

T11 and T13 are the ones that matter. Every other test constructs its own
library and would keep passing if the wiring were reverted; T11 goes through
``roofline._profile_for`` and T13 through the real ``_pp_cut_card_rates``.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from sglang.srt.planner import rate_env as renv
from sglang.srt.planner.card_library import (
    CardCapacityMismatch,
    CardLibrary,
    CardSpec,
)

#: What the driver calls both variants.
DRIVER_3080 = "NVIDIA GeForce RTX 3080"
#: What this rig's cards actually measure (MEMORY: 2x RTX 3080 20 GB).
RIG_3080_MIB = 20480
#: What the seed entry the name resolves to claims.
SEED_3080_MIB = 10240


class CapacityCollisionTest(unittest.TestCase):
    """T1-T5. A name that cannot select a variant must not pick one anyway."""

    def test_t1_the_collision_exists_in_the_seed_set(self):
        """The premise, pinned so a later seed edit cannot quietly void it."""
        lib = CardLibrary()
        self.assertEqual(lib.get("RTX 3080").total_mib, SEED_3080_MIB)
        self.assertEqual(lib.get("RTX 3080 20GB").total_mib, RIG_3080_MIB)
        # And the driver's name reaches only the first of them.
        self.assertEqual(lib.get(DRIVER_3080).total_mib, SEED_3080_MIB)

    def test_t2_resolve_selects_the_variant_the_measurement_names(self):
        """The fix's happy path: the library HAS the right card, so use it."""
        lib = CardLibrary()
        spec = lib.resolve(DRIVER_3080, total_mib=RIG_3080_MIB)
        self.assertEqual(spec.name, "RTX 3080 20GB")
        self.assertEqual(spec.total_mib, RIG_3080_MIB)

    def test_t3_resolve_still_finds_the_10gb_card_when_that_is_the_card(self):
        lib = CardLibrary()
        spec = lib.resolve(DRIVER_3080, total_mib=SEED_3080_MIB)
        self.assertEqual(spec.name, "RTX 3080")

    def test_t4_no_matching_variant_refuses_and_names_both_numbers(self):
        """A capacity in the library for neither variant is a refusal, not a
        nearest-neighbour pick."""
        lib = CardLibrary()
        with self.assertRaises(CardCapacityMismatch) as caught:
            lib.resolve(DRIVER_3080, total_mib=16384)
        message = str(caught.exception)
        self.assertIn("16384", message)
        self.assertIn(str(SEED_3080_MIB), message)
        self.assertIn(str(RIG_3080_MIB), message)

    def test_t5_nvml_rounding_stays_inside_the_tolerance(self):
        """NVML totals are not the nominal number: the seed 5090 is 32607, not
        32768. A few hundred MiB of carve-out must not read as a variant
        mismatch."""
        lib = CardLibrary()
        spec = lib.resolve("NVIDIA GeForce RTX 5090", total_mib=32510)
        self.assertEqual(spec.name, "RTX 5090")

    def test_t5b_an_exact_name_match_beats_a_sibling_model(self):
        """``RTX 3090`` and ``RTX 3090 Ti`` are 12 MiB apart, so a capacity
        filter alone cannot tell them apart. The exact name must win."""
        lib = CardLibrary()
        self.assertEqual(lib.resolve("RTX 3090", total_mib=24576).name, "RTX 3090")
        self.assertEqual(
            lib.resolve("RTX 3090 Ti", total_mib=24564).name, "RTX 3090 Ti"
        )

    def test_t5c_an_unknown_card_is_still_a_plain_key_error(self):
        lib = CardLibrary()
        with self.assertRaises(KeyError):
            lib.resolve("Voodoo 2", total_mib=12)


class MeasuredWinsOverSeedTest(unittest.TestCase):
    """T6-T7. The measurement is the authority on the card it measured."""

    def test_t6_a_measured_entry_outranks_the_seed_of_the_same_name(self):
        lib = CardLibrary()
        lib.add(
            CardSpec(
                name="RTX 3080",
                total_mib=RIG_3080_MIB,
                gemm_tflops=50.81,
                membw_gbs=717.4,
                source="measured",
            ),
            overwrite=True,
        )
        spec = lib.resolve(DRIVER_3080, total_mib=RIG_3080_MIB)
        self.assertEqual(spec.source, "measured")
        self.assertEqual(spec.total_mib, RIG_3080_MIB)
        self.assertEqual(spec.gemm_tflops, 50.81)

    def test_t7_a_measured_entry_that_disagrees_is_still_refused(self):
        """An artifact carried from another host does not get to redefine the
        card in this one."""
        lib = CardLibrary()
        lib.add(
            CardSpec(
                name="RTX 3080",
                total_mib=SEED_3080_MIB,
                gemm_tflops=50.81,
                membw_gbs=717.4,
                source="measured",
            ),
            overwrite=True,
        )
        with self.assertRaises(CardCapacityMismatch):
            lib.resolve(DRIVER_3080, total_mib=RIG_3080_MIB)


class RateEnvironmentFingerprintTest(unittest.TestCase):
    """T8-T10. A rate without its environment is a rate nobody can date."""

    def test_t8_the_token_round_trips(self):
        env = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)
        self.assertEqual(renv.RateEnv.parse(env.token), env)

    def test_t9_a_power_limit_change_reads_as_stale(self):
        """The 2026-08-05 cut, in the units NVML reports it."""
        before = renv.RateEnv(driver_version="580.65.06", power_limit_mw=320000)
        after = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)
        verdict = renv.check_rate_freshness(before.token, [after])
        self.assertTrue(verdict.stale)
        self.assertIn("320", verdict.reason)
        self.assertIn("200", verdict.reason)
        # ... and the same environment is fresh.
        self.assertTrue(renv.check_rate_freshness(after.token, [after]).fresh)

    def test_t9b_a_driver_change_reads_as_stale(self):
        before = renv.RateEnv(driver_version="570.00.00", power_limit_mw=200000)
        after = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)
        self.assertTrue(renv.check_rate_freshness(before.token, [after]).stale)

    def test_t10_no_fingerprint_is_stale_unknown_and_loud(self):
        """Backward-readable: an artifact written before fingerprinting still
        loads, and says so."""
        now = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)
        for token in (None, ""):
            with self.subTest(token=token):
                verdict = renv.check_rate_freshness(token, [now])
                self.assertEqual(verdict.state, "unknown")
                self.assertFalse(verdict.fresh)
                self.assertTrue(verdict.reason)

    def test_t10b_an_unreadable_nvml_cannot_claim_staleness(self):
        env = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)
        verdict = renv.check_rate_freshness(env.token, [])
        self.assertEqual(verdict.state, "unknown")
        self.assertFalse(verdict.stale)

    def test_t10c_the_fingerprint_survives_a_library_save_load(self):
        import os
        import tempfile

        env = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)
        lib = CardLibrary()
        lib.add(
            CardSpec(
                name="RTX 3080 20GB",
                total_mib=RIG_3080_MIB,
                gemm_tflops=50.81,
                membw_gbs=717.4,
                source="measured",
                rate_env=env.token,
            ),
            overwrite=True,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "card_library.json")
            lib.save(path)
            back = CardLibrary.load(path)
            self.assertEqual(back.get("RTX 3080 20GB").rate_env, env.token)
            self.assertEqual(back.get("RTX 3080 20GB").source, "measured")
            # A pre-fingerprint file still loads, and reads as unknown.
            with open(path) as f:
                data = json.load(f)
            for entry in data["profiles"]:
                entry.pop("rate_env", None)
                entry.pop("source", None)
            with open(path, "w") as f:
                json.dump(data, f)
            old = CardLibrary.load(path)
            self.assertIsNone(old.get("RTX 3080 20GB").rate_env)
            self.assertEqual(old.get("RTX 3080 20GB").source, "seed")


class ConsumersDetectTest(unittest.TestCase):
    """T11-T13. The wiring, not the helper."""

    def test_t11_the_roofline_declines_to_price_the_wrong_card(self):
        """``_profile_for`` is the roofline's name-keyed fallback. Handed a
        20480 MiB card it must not return the 10240 MiB seed."""
        from sglang.srt.planner import roofline

        lib = CardLibrary()
        wrong = roofline._profile_for(lib, DRIVER_3080, total_mib=RIG_3080_MIB)
        self.assertIsNotNone(wrong, "the library carries the 20 GB variant")
        self.assertEqual(wrong.total_mib, RIG_3080_MIB)

        # A capacity no variant matches: no profile at all, rather than one
        # that would price the card from another card's nameplate.
        self.assertIsNone(roofline._profile_for(lib, DRIVER_3080, total_mib=16384))

    def test_t12_the_roofline_call_sites_pass_the_measured_total(self):
        """The wiring test. ``_profile_for`` growing a parameter no caller
        supplies would leave the defect exactly where it was."""
        import inspect

        from sglang.srt.planner import roofline

        source = inspect.getsource(roofline)
        calls = [
            line.strip()
            for line in source.splitlines()
            if "_profile_for(" in line and "def _profile_for" not in line
        ]
        self.assertTrue(calls, "no _profile_for call sites found")
        for call in calls:
            with self.subTest(call):
                self.assertIn("total_mib", call)

    def test_t13_pp_solve_cut_refuses_a_rate_from_another_power_limit(self):
        """Through the real handler: a library on disk whose rate was measured
        at 320 W, on a rig now reporting 200 W."""
        import os
        import tempfile

        from sglang.srt.server_args import ServerArgs

        before = renv.RateEnv(driver_version="580.65.06", power_limit_mw=320000)
        now = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)

        lib = CardLibrary()
        lib.add(
            CardSpec(
                name="RTX 3080 20GB",
                total_mib=RIG_3080_MIB,
                gemm_tflops=65.57,
                membw_gbs=717.4,
                source="measured",
                rate_env=before.token,
            ),
            overwrite=True,
        )
        args = ServerArgs.__new__(ServerArgs)
        args.pp_size = 1

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "card_library.json")
            lib.save(path)
            with mock.patch.dict(os.environ, {"SGLANG_CARD_LIBRARY": path}):
                with mock.patch.object(
                    renv, "current_envs_by_name", return_value={"rtx 3080": [now]}
                ):
                    with self.assertRaises(ValueError) as caught:
                        args._pp_cut_card_rates([DRIVER_3080])
        message = str(caught.exception)
        self.assertIn("card_rate_pass --run", message)
        self.assertIn("320", message)
        self.assertIn("200", message)


    def test_t14_the_pass_show_dates_every_rate_it_prints(self):
        """``--show`` is the operator's view of the store. A rate printed
        without its freshness is a rate presented as current."""
        import io
        import os
        import tempfile
        from contextlib import redirect_stdout

        from sglang.srt.planner import card_rate_pass as crp

        before = renv.RateEnv(driver_version="580.65.06", power_limit_mw=320000)
        now = renv.RateEnv(driver_version="580.65.06", power_limit_mw=200000)
        lib = CardLibrary()
        lib.add(
            CardSpec(
                name="RTX 3080 20GB",
                total_mib=RIG_3080_MIB,
                gemm_tflops=65.57,
                membw_gbs=717.4,
                source="measured",
                rate_env=before.token,
            ),
            overwrite=True,
        )
        # A second card whose rate predates fingerprinting entirely.
        lib.add(
            CardSpec(
                name="RTX 5090",
                total_mib=32607,
                gemm_tflops=203.57,
                membw_gbs=1661.7,
                source="measured",
            ),
            overwrite=True,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "card_library.json")
            lib.save(path)
            with mock.patch.object(
                renv, "current_envs_by_name", return_value={"rtx 3080 20gb": [now]}
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = crp.main(["--show", "--path", path])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("STALE", out)
        self.assertIn("UNKNOWN", out)
        for line in out.splitlines():
            if "gemm" in line:
                self.assertRegex(line, r"\[(FRESH|STALE|UNKNOWN)\]")


if __name__ == "__main__":
    unittest.main()
