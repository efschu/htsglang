# SPDX-License-Identifier: Apache-2.0
"""#656: the solver's margin carries a MEASURED corridor shortfall.

WHY. ``seam_allowed_tokens`` targets equality, so the sizer's own margin
(``DEFAULT_MARGIN_MIB``) is the measurement error bar and nothing more. It
guarantees the seam is fundable AT REST and says nothing about the level the
card reaches while the seam RUNS. On this rig the #656 acceptance measured
that drawdown at 1814-1852 MiB against a gate that assumes 512, and five
cutovers entered through a gate with no objection and took the card to 886
MiB -- 138 below the law.

That shortfall is not derivable from the geometry: it is a property of the
geometry AND the load AND the card. So the honest term is the one the
runtime writes down after it has seen one, carried to the next boot by the
two-boot protocol the seam record already implements -- and it is ZERO until
a breach is actually observed, which is what stops it becoming another
constant that travels between rigs.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers import phase_flip_seam_reserve as seam
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20


class _RecordDir:
    """Point record_path at a temp file for the duration of a test."""

    def __init__(self, testcase):
        self.dir = tempfile.TemporaryDirectory()
        testcase.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "seam-rank1.json")

    def patch(self):
        return mock.patch.object(seam, "record_path", lambda *a, **k: self.path)

    def write(self, **fields):
        payload = {
            "fixed_bytes": 1024,
            "per_row_bytes": 2.0,
            "have_bytes": 4096 * MIB,
            "id_space": 1000,
            "detail": "test",
        }
        payload.update(fields)
        with open(self.path, "w") as fh:
            json.dump(payload, fh)

    def read(self):
        with open(self.path) as fh:
            return json.load(fh)


class TestTheMarginHasTwoTerms(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.pop(seam.ENV_MARGIN_MIB, None)

    def tearDown(self):
        os.environ.pop(seam.ENV_MARGIN_MIB, None)
        if self._saved is not None:
            os.environ[seam.ENV_MARGIN_MIB] = self._saved

    def test_no_measurement_is_byte_identical_to_before(self):
        """The whole safety argument: a rig that never breached is sized
        exactly as it was."""
        self.assertEqual(seam.seam_margin_bytes(), seam.DEFAULT_MARGIN_MIB << 20)
        self.assertEqual(
            seam.seam_margin_bytes(seam.SeamReserve()),
            seam.DEFAULT_MARGIN_MIB << 20,
        )

    def test_a_measured_shortfall_is_added_to_the_error_bar(self):
        reserve = seam.SeamReserve(corridor_shortfall_bytes=138 * MIB)
        self.assertEqual(
            seam.seam_margin_bytes(reserve),
            (seam.DEFAULT_MARGIN_MIB << 20) + 138 * MIB,
        )

    def test_the_env_override_still_pins_the_whole_margin(self):
        os.environ[seam.ENV_MARGIN_MIB] = "384"
        reserve = seam.SeamReserve(corridor_shortfall_bytes=138 * MIB)
        self.assertEqual(seam.seam_margin_bytes(reserve), 384 * MIB)

    def test_a_malformed_override_falls_back_to_the_derived_margin(self):
        """It must not fall back to ZERO -- a zero-margin pool is the exact
        failure the margin exists to prevent."""
        reserve = seam.SeamReserve(corridor_shortfall_bytes=138 * MIB)
        for bad in ("not-a-number", "-1"):
            os.environ[seam.ENV_MARGIN_MIB] = bad
            self.assertEqual(
                seam.seam_margin_bytes(reserve),
                (seam.DEFAULT_MARGIN_MIB << 20) + 138 * MIB,
                f"override {bad!r}",
            )

    def test_the_margin_shrinks_the_pool_it_sizes(self):
        """The term has to actually reach the solver, not just exist."""
        reserve = seam.SeamReserve(
            fixed_bytes=64 * MIB,
            per_row_bytes=64.0,
            have_bytes=4096 * MIB,
            id_space=100000,
        )
        wider = seam.SeamReserve(
            fixed_bytes=64 * MIB,
            per_row_bytes=64.0,
            have_bytes=4096 * MIB,
            id_space=100000,
            corridor_shortfall_bytes=138 * MIB,
        )
        cell = 8192
        base = seam.seam_allowed_tokens(cell, reserve)
        with_shortfall = seam.seam_allowed_tokens(cell, wider)
        self.assertLess(with_shortfall, base)


class TestTheRecord(CustomTestCase):
    def test_a_shortfall_is_recorded_and_read_back(self):
        rec = _RecordDir(self)
        with rec.patch():
            rec.write()
            self.assertEqual(
                seam.record_corridor_shortfall(SimpleNamespace(), 1, 138 * MIB),
                138 * MIB,
            )
            self.assertEqual(rec.read()["corridor_shortfall_bytes"], 138 * MIB)
            back = seam.read_seam_reserve(SimpleNamespace(), 1)
            self.assertEqual(back.corridor_shortfall_bytes, 138 * MIB)

    def test_it_is_a_monotonic_maximum(self):
        """A shallower breach later does not mean the deeper one cannot
        recur, so the worst instant ever seen is what the pool is sized for
        -- the same rule the corridor law itself is stated with."""
        rec = _RecordDir(self)
        with rec.patch():
            rec.write()
            seam.record_corridor_shortfall(SimpleNamespace(), 1, 138 * MIB)
            seam.record_corridor_shortfall(SimpleNamespace(), 1, 40 * MIB)
            self.assertEqual(rec.read()["corridor_shortfall_bytes"], 138 * MIB)
            seam.record_corridor_shortfall(SimpleNamespace(), 1, 200 * MIB)
            self.assertEqual(rec.read()["corridor_shortfall_bytes"], 200 * MIB)

    def test_a_cold_rank_with_no_record_writes_nothing(self):
        """Inventing a record here would hand the next boot a seam floor
        nobody measured."""
        rec = _RecordDir(self)
        with rec.patch():
            self.assertIsNone(
                seam.record_corridor_shortfall(SimpleNamespace(), 1, 138 * MIB)
            )
            self.assertFalse(os.path.exists(rec.path))

    def test_zero_and_negative_shortfalls_are_ignored(self):
        rec = _RecordDir(self)
        with rec.patch():
            rec.write()
            self.assertIsNone(seam.record_corridor_shortfall(SimpleNamespace(), 1, 0))
            self.assertIsNone(seam.record_corridor_shortfall(SimpleNamespace(), 1, -5))
            self.assertEqual(rec.read().get("corridor_shortfall_bytes", 0), 0)

    def test_a_re_measured_seam_does_not_discard_the_shortfall(self):
        """write_seam_reserve runs at the END of a flip boot and the
        shortfall is written mid-run by a different event. A boot that
        re-measures its seam must not throw away a breach an earlier boot
        paid to learn about."""
        rec = _RecordDir(self)
        with rec.patch():
            rec.write()
            seam.record_corridor_shortfall(SimpleNamespace(), 1, 138 * MIB)
            seam.write_seam_reserve(
                SimpleNamespace(),
                1,
                fixed_bytes=2048,
                per_row_bytes=3.0,
                detail="re-measured",
                have_bytes=5000 * MIB,
                id_space=2000,
            )
            after = rec.read()
            self.assertEqual(after["corridor_shortfall_bytes"], 138 * MIB)
            self.assertEqual(after["fixed_bytes"], 2048)

    def test_a_record_written_before_656_reads_as_never_measured(self):
        rec = _RecordDir(self)
        with rec.patch():
            rec.write()  # no corridor_shortfall_bytes key at all
            back = seam.read_seam_reserve(SimpleNamespace(), 1)
            self.assertEqual(back.corridor_shortfall_bytes, 0)
            self.assertEqual(
                seam.seam_margin_bytes(back), seam.DEFAULT_MARGIN_MIB << 20
            )


if __name__ == "__main__":
    unittest.main()
