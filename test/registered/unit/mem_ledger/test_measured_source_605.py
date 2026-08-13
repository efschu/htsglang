"""#605 stage 4: the boot history may calibrate a term only where it is stable.

THE FAILURE THIS PINS is not "the measurement is wrong". It is the far more
tempting one: taking a median of a post that ranges over 2408 MiB between boots
of an unchanged config and installing it as a calibrated constant, because a
median always exists and looks like an answer. That converts variance into
false precision and is the guessing game wearing a lab coat.

Measured over the 14 ship boots of 2026-08-13, the two populations are:
``cuda_context_and_comm`` at spread EXACTLY 0 (888 / 482 / 482 MiB), and
``kv_pool_target`` at spread 1364-2408 MiB. The first may calibrate a residual.
The second may not, and the module must SAY SO rather than fall silent.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.mem_ledger import measured
from sglang.srt.mem_ledger.measured import (
    MIB,
    MIN_BOOTS,
    MeasuredPost,
    describe,
    residual_overrides,
)


def _write_boots(directory, per_boot_context_bytes, uuid="GPU-aaaa", pid=4001):
    """One rank, one card, N boots, each with a process_start + pre_weight_load."""
    path = os.path.join(directory, "flight_marks_rank0.jsonl")
    with open(path, "w") as handle:
        for index, context in enumerate(per_boot_context_bytes):
            for phase, resident in (
                ("process_start", 0),
                ("pre_weight_load", context),
            ):
                handle.write(
                    json.dumps(
                        {
                            "phase": phase,
                            "rank": 0,
                            "boot_id": f"boot{index}",
                            "pid": pid,
                            "wall": 1000.0 + index * 10 + (0 if resident == 0 else 1),
                            "monotonic": float(
                                index * 10 + (0 if resident == 0 else 1)
                            ),
                            "card_uuid": uuid,
                            "nvml_self_bytes": resident,
                            "reserved_bytes": 0,
                        }
                    )
                    + "\n"
                )


class TestStablePostCalibrates(unittest.TestCase):
    def test_a_context_constant_over_many_boots_becomes_an_override(self):
        with tempfile.TemporaryDirectory() as directory:
            # The 5090's measured reality: 888 MiB, every boot, no spread.
            _write_boots(directory, [888 * MIB] * 14)
            overrides = residual_overrides(directory, force=True)

        self.assertEqual(overrides, {"GPU-aaaa": {"cuda_context_bytes": 888 * MIB}})


class TestWidePostIsRefused(unittest.TestCase):
    def test_a_post_ranging_over_gigabytes_does_not_become_a_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            # The shape of kv_pool_target: a real measurement, a useless
            # constant. A median exists; it must not be installed.
            _write_boots(
                directory,
                [(4312 + step * 100) * MIB for step in range(14)],
            )
            overrides = residual_overrides(directory, force=True)

        self.assertEqual(
            overrides,
            {},
            "a post with a 1300 MiB spread was installed as a calibrated "
            "constant; the spread is the reason it must not be",
        )

    def test_the_refusal_states_its_reason_rather_than_falling_silent(self):
        wide = MeasuredPost(
            card_uuid="GPU-aaaa",
            post="kv_pool_target",
            values_bytes=tuple((4312 + s * 100) * MIB for s in range(14)),
        )
        self.assertFalse(wide.stable)
        self.assertIn("spread", wide.why_not())
        self.assertIn("1300 MiB", wide.why_not())


class TestTooFewBootsIsNotACalibration(unittest.TestCase):
    def test_agreement_across_two_boots_is_a_coincidence_not_a_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_boots(directory, [888 * MIB] * (MIN_BOOTS - 1))
            self.assertEqual(residual_overrides(directory, force=True), {})

        with tempfile.TemporaryDirectory() as directory:
            _write_boots(directory, [888 * MIB] * MIN_BOOTS)
            self.assertEqual(
                residual_overrides(directory, force=True),
                {"GPU-aaaa": {"cuda_context_bytes": 888 * MIB}},
            )


class TestOffByDefault(unittest.TestCase):
    """The gate is the env var; without it the ledger sees nothing at all."""

    def setUp(self):
        self._saved = os.environ.pop(measured.MEASURED_ENV, None)

    def tearDown(self):
        if self._saved is not None:
            os.environ[measured.MEASURED_ENV] = self._saved
        else:
            os.environ.pop(measured.MEASURED_ENV, None)

    def test_without_the_env_var_no_override_is_produced(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_boots(directory, [888 * MIB] * 14)
            self.assertEqual(
                residual_overrides(directory),
                {},
                "the measured source must be inert unless it is switched on",
            )

    def test_with_the_env_var_the_same_history_does_produce_one(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_boots(directory, [888 * MIB] * 14)
            os.environ[measured.MEASURED_ENV] = "1"
            self.assertEqual(
                residual_overrides(directory),
                {"GPU-aaaa": {"cuda_context_bytes": 888 * MIB}},
            )


class TestDescribeShowsBothPopulations(unittest.TestCase):
    def test_a_declined_post_is_visible_in_the_table(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_boots(directory, [888 * MIB, 900 * MIB, 888 * MIB])
            text = describe(directory)
        self.assertIn("DECLINED", text)
        self.assertIn("GPU-aaaa", text)

    def test_no_history_is_reported_as_no_history_not_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIn("nothing measured", describe(directory))


if __name__ == "__main__":
    unittest.main()
