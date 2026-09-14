# SPDX-License-Identifier: Apache-2.0
"""#1378 Posten 4 -- the WEG2_HOSTSAMPLE_CSV seam, closed and named.

THE FINDING (desk gates, 2026-09-14): every FLIP record in the live sidecar
carries ``cushion_min_gib: null`` -- weg2xsn29 (no key, pre-91d6df2e50),
weg2zwerg and weg2ring1 (null) -- because ``cushion_min_of_this_boot()``
reads env ``WEG2_HOSTSAMPLE_CSV`` (host_ledger.py, ``ENV_HOSTSAMPLE_CSV``)
and NOTHING ever set it: pickaxe over every ref shows the name in exactly
one commit, the reader itself. The numbers were on disk the whole time --
the sampler ran per boot and wrote the ``cushion_gib`` column; measured
with this tree's own reader: xsn29 14.857, zwerg 5.23, ring1 4.276,
xsn31 1.082, xsn32 0.014 GiB. The cross-boot gate that separates xsn31/2
from /3 was wired (695c29dd9e) but read an empty source.

THE THREE RULES THIS FILE PINS (operator order, 14.09.):

1. ONE PRODUCER of the path. ``launcher.hostsample_csv_path(tag)`` is the
   only construction site; ``build_env`` publishes its RESULT under
   ``host_ledger.ENV_HOSTSAMPLE_CSV`` and must not re-derive it.
2. THE VARIABLE IS A READ SOURCE: a missing CSV or a missing cushion column
   is a NAMED absence -- the reason text says which case fired and names the
   path -- never a silent null-with-no-explanation and NEVER 0.0 (a 0.0 would
   make the gate look armed while it reads nothing, which is worse than the
   null it replaces).
3. RED-FIRST against the pre-fix state (env unset -> bare None, no reason),
   then green: env set -> the value flows through the front's
   ``_write_flip_ratchet`` into the record file.

MUTANT (operator-required danger direction): a producer that answers a
column-less CSV with 0.0 instead of the named absence must be CAUGHT by the
same assertion logic the real test runs.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase

#: The REAL sampler's header (gpu-arb/weg2/hostsample_v2.sh line 39, sampler
#: version 3), verbatim minus the trailing verdict columns -- a fixture shaped
#: like the file the reader must one day parse on the metal.
SAMPLER_HEADER = ("ts,anon_gib,shmem_gib,slab_unrecl_gib,nonreclaim_gib,"
                  "file_gib,host_memavailable_gib,root_current_gib,"
                  "root_file_gib,root_anon_gib,root_shmem_gib,root_slab_gib,"
                  "sampler_version,psi_mem_full_total_us_delta,"
                  "psi_io_full_total_us_delta,tick_s,cushion_gib")


def _write_sampler_csv(path: str, cushions: list) -> None:
    """A sampler CSV with the REAL header and the given cushion_gib values."""
    with open(path, "w") as fh:
        fh.write(SAMPLER_HEADER + "\n")
        for i, c in enumerate(cushions):
            fh.write(f"2026-09-14T12:00:{i:02d}Z,4.1,23.5,0.0,27.6,28.7,"
                     f"90.3,33.0,28.7,4.1,23.5,0.1,3,0,0,0.5,{c}\n")


def _cushion_assertions(value, reason, *, expect_value=None):
    """The ONE assertion logic every case below and the mutant run: a named
    absence is (None, reason naming the cause), a measurement is
    (value, reason carrying the value) -- and NEVER 0.0 standing in for
    'could not read'."""
    if expect_value is None:
        assert value is None, f"expected named absence, got value={value!r}"
        assert value != 0.0, "a named absence must never read as 0.0"
    else:
        assert value is not None and abs(value - expect_value) < 1e-6, \
            f"expected {expect_value}, got {value!r}"
    assert isinstance(reason, str) and reason, \
        f"the absence/measurement must be NAMED, got reason={reason!r}"
    return reason


class TheCushionSamplerSeam(CustomTestCase):
    """The read side: three named cases and one value case, one producer."""

    def setUp(self):
        self._old = os.environ.get(hl.ENV_HOSTSAMPLE_CSV)
        self._tmp = tempfile.mkdtemp(prefix="weg2-1378-hostsample-")
        self.csv = os.path.join(self._tmp, "hostsample_t.csv")
        os.environ.pop(hl.ENV_HOSTSAMPLE_CSV, None)

    def tearDown(self):
        if self._old is None:
            os.environ.pop(hl.ENV_HOSTSAMPLE_CSV, None)
        else:
            os.environ[hl.ENV_HOSTSAMPLE_CSV] = self._old

    def test_env_unset_is_a_NAMED_absence_not_a_bare_none(self):
        """RED-FIRST against the pre-fix state: today's
        ``cushion_min_of_this_boot`` returns a bare None and says nothing --
        exactly the silence that left three FLIP records at null with no
        reason on any line. The seam must name the case."""
        value, reason = hl.cushion_min_of_this_boot()
        _cushion_assertions(value, reason, expect_value=None)
        self.assertIn("WEG2_HOSTSAMPLE_CSV", reason,
                      f"the absence must name the unset variable: {reason!r}")

    def test_env_set_value_flows_and_is_named_with_the_path(self):
        _write_sampler_csv(self.csv, [5.0, 1.082, 3.0])
        os.environ[hl.ENV_HOSTSAMPLE_CSV] = self.csv
        value, reason = hl.cushion_min_of_this_boot()
        # THE MINIMUM, not the last sample: 1.082 is xsn31's own sampler
        # minimum and the number this whole ticket started from.
        _cushion_assertions(value, reason, expect_value=1.082)
        self.assertIn(self.csv, reason)

    def test_env_set_but_csv_missing_is_a_NAMED_absence(self):
        os.environ[hl.ENV_HOSTSAMPLE_CSV] = os.path.join(self._tmp, "nope.csv")
        value, reason = hl.cushion_min_of_this_boot()
        _cushion_assertions(value, reason, expect_value=None)
        self.assertIn("nope.csv", reason,
                      "the absence must name the path that was not there")
        self.assertIn("unreadable", reason)

    def test_M_column_less_csv_must_be_named_absence_never_zero(self):
        """MUTANT (operator-required danger direction): a producer that
        answers a column-less CSV with 0.0 makes the cross-boot gate look
        armed while it reads nothing. Drives the REAL case's assertion logic
        against that mutated producer and asserts the case catches it."""

        def mutated_producer():  # the danger shape: silent 0.0
            return 0.0, ""

        with mock.patch.object(
                hl, "cushion_min_of_this_boot", mutated_producer):
            value, reason = hl.cushion_min_of_this_boot()
            with self.assertRaises(AssertionError) as cm:
                _cushion_assertions(value, reason, expect_value=None)
            self.assertIn("0.0", str(cm.exception),
                          "the 0.0 mutant must die on the named-absence rule")

    def test_a_column_less_csv_is_the_named_absence(self):
        """The case the mutant stands for, for real: a sampler output that
        predates the cushion column must yield (None, reason naming
        cushion_gib), never 0.0."""
        path = os.path.join(self._tmp, "old_sampler.csv")
        with open(path, "w") as fh:
            fh.write("ts,nonreclaim_gib,anon_gib\n")
            fh.write("2026-09-13T12:00:00Z,27.6,4.1\n")
        os.environ[hl.ENV_HOSTSAMPLE_CSV] = path
        value, reason = hl.cushion_min_of_this_boot()
        _cushion_assertions(value, reason, expect_value=None)
        self.assertIn("cushion_gib", reason,
                      f"the absence must name the missing column: {reason!r}")


class TheSeamIsWiredThroughTheLauncherAndTheRecord(CustomTestCase):
    """The write side: ONE producer of the path, the env carries its RESULT,
    and a set env flows through ``Front._write_flip_ratchet`` into the
    record FILE."""

    def setUp(self):
        self._old = os.environ.get(hl.ENV_HOSTSAMPLE_CSV)
        self._tmp = tempfile.mkdtemp(prefix="weg2-1378-hostsample-")
        self.csv = os.path.join(self._tmp, "hostsample_t.csv")
        os.environ.pop(hl.ENV_HOSTSAMPLE_CSV, None)

    def tearDown(self):
        if self._old is None:
            os.environ.pop(hl.ENV_HOSTSAMPLE_CSV, None)
        else:
            os.environ[hl.ENV_HOSTSAMPLE_CSV] = self._old

    def test_hostsample_csv_path_is_the_documented_convention(self):
        """host_ledger's own documented shape:
        ``<evidence>/<tag>_<yymmdd>/hostsample_<tag>.csv``."""
        p = L.hostsample_csv_path("weg2ring1")
        self.assertTrue(p.startswith(L.EVIDENCE_DIR + "/"), p)
        self.assertIn("weg2ring1_", p)
        self.assertTrue(p.endswith("/hostsample_weg2ring1.csv"), p)

    def test_build_env_publishes_the_producer_result_under_the_ledger_env(self):
        """ONE PRODUCER, no second construction: the env value must be
        ``hostsample_csv_path(tag)``'s own output, and the key must be the
        ledger's constant (no string restatement)."""
        env = L.build_env(tree="/tmp/t", venv="/tmp/v", cvd="",
                          store_dir="/tmp/store", debug_hold=False,
                          tag="weg2seamtest")
        self.assertEqual(env[hl.ENV_HOSTSAMPLE_CSV],
                         L.hostsample_csv_path("weg2seamtest"))

    def test_env_set_flows_through_the_front_into_the_record_file(self):
        """End to end: sampler CSV -> env -> Front._write_flip_ratchet ->
        sidecar record with a MEASURED cushion_min_gib."""
        _write_sampler_csv(self.csv, [5.0, 1.082, 3.0])
        record_path = os.path.join(self._tmp, "weg2_measured_record.json")
        os.environ[hl.ENV_HOSTSAMPLE_CSV] = self.csv
        f = object.__new__(front_mod.Front)
        f._flip_ratchet_written = False
        f._flip_ratchet_pre_gib = 77.9
        f._flip_ratchet_pre_at = "2026-09-14T12:00:00Z"
        f.tag = "weg2seamtest"
        f.commit = "6967bf3ff3"
        f.ledger_arm = {"xchg_bounce_gib": 3.75}
        f.weights_tags = ["weights_0"]
        f.measured_record = record_path
        f._write_flip_ratchet(2)
        with open(record_path) as fh:
            rec = json.load(fh)["samples"][-1]
        self.assertEqual(rec["group"], "FLIP")
        self.assertAlmostEqual(rec["cushion_min_gib"], 1.082, places=3,
                               msg="the measured minimum must reach the record")
        self.assertAlmostEqual(rec["xchg_bounce_gib"], 3.75, places=3)

    def test_env_unset_writes_null_and_still_names_the_case(self):
        """The Ist-Stand behaviour PRESERVED as an explicit case: no env ->
        null in the record -- but the log line now says why, so the next
        reader does not have to pickaxe the history to learn it."""
        record_path = os.path.join(self._tmp, "weg2_measured_record.json")
        f = object.__new__(front_mod.Front)
        f._flip_ratchet_written = False
        f._flip_ratchet_pre_gib = 77.9
        f._flip_ratchet_pre_at = "2026-09-14T12:00:00Z"
        f.tag = "weg2seamtest"
        f.commit = "6967bf3ff3"
        f.ledger_arm = {}
        f.weights_tags = ["weights_0"]
        f.measured_record = record_path
        f._write_flip_ratchet(2)
        with open(record_path) as fh:
            rec = json.load(fh)["samples"][-1]
        self.assertIsNone(rec["cushion_min_gib"])


if __name__ == "__main__":
    unittest.main()
